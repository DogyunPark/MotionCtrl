import argparse
import datetime
import inspect
import os, sys
from omegaconf import OmegaConf
import json
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, save_videos_jpg
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np
from motionctrl.modified_modules import (
    Adapted_TemporalTransformerBlock_forward, unet3d_forward,
    Adapted_CrossAttnDownBlock3D_forward, Adapted_DownBlock3D_forward)

from motionctrl.adapter import Adapter
from motionctrl.utils.util import instantiate_from_config
from motionctrl.util import get_traj_features, get_batch_motion, get_opt_from_video, vis_opt_flow

sys.path.append("/home/dogyun/CameraCtrl/")
from cameractrl.data.dataset import Camera
#from cameractrl.models.unet import UNet3DConditionModel
#from cameractrl.pipelines.pipeline_animation import AnimationPipeline

@torch.no_grad()
def main(args):
    os.makedirs(args.out_root, exist_ok=True)
    video_pth = '{}/video'.format(args.out_root)
    image_pth = '{}/image'.format(args.out_root)
    os.makedirs(video_pth, exist_ok=True)
    os.makedirs(image_pth, exist_ok=True)

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}" #-{time_str}"

    config  = OmegaConf.load(args.config)

    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)
        savedir = f"{savedir}_H{model_config.H}_W{model_config.W}"

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # print(f"Setting the attention processors")
        # unet.set_all_attn_processor(add_spatial_lora=True,
        #                             add_motion_lora=False,
        #                             lora_kwargs={"lora_rank": 2, "lora_scale": 1.0},
        #                             motion_lora_kwargs={"lora_rank": -1, "lora_scale": 1.0},
        # )

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()


        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ) #.to("cuda")

        # image_lora_ckpt = "models/Motion_Module/RealEstate10K_LoRA.ckpt"
        # if image_lora_ckpt is not None:
        #     print(f"Loading the lora checkpoint from {image_lora_ckpt}")
        #     lora_checkpoints = torch.load(image_lora_ckpt, map_location=unet.device)
        #     if 'lora_state_dict' in lora_checkpoints.keys():
        #         lora_checkpoints = lora_checkpoints['lora_state_dict']
        #     _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        #     assert len(lora_u) == 0
        #     print(f'Loading done')

        if model_config.get("dreambooth_path", "") != "":
            savedir += "_dreambooth"

        bound_moudule = unet3d_forward.__get__(unet, unet.__class__)
        setattr(unet, "forward", bound_moudule)

        # motionctrl
        cmcm_checkpoint_path       = model_config.get("cmcm_checkpoint_path", "")
        omcm_checkpoint_path       = model_config.get("omcm_checkpoint_path", "")
        optical_flow_config        = model_config.get("optical_flow_config", None)
        if optical_flow_config is not None:
            use_optical_flow = True
        else:
            use_optical_flow = False
        # import pdb; pdb.set_trace()

        if cmcm_checkpoint_path != "" and os.path.exists(cmcm_checkpoint_path):
            name_part = cmcm_checkpoint_path.split('/')
            savedir = savedir + f"_cmcm"

            for _name, _module in unet.named_modules():
                if _module.__class__.__name__ == "TemporalTransformerBlock":
                    bound_moudule = Adapted_TemporalTransformerBlock_forward.__get__(_module, _module.__class__)
                    setattr(_module, "forward", bound_moudule)

                    cc_projection = nn.Linear(_module.attention_blocks[-1].to_k.in_features + 12, _module.attention_blocks[-1].to_k.in_features)
                    nn.init.eye_(list(cc_projection.parameters())[0][:_module.attention_blocks[-1].to_k.in_features, :_module.attention_blocks[-1].to_k.in_features])
                    nn.init.zeros_(list(cc_projection.parameters())[1])
                    cc_projection.requires_grad_(True)

                    _module.add_module('cc_projection', cc_projection)

            # load cmcm checkpoint
            print(f"load cmcm from {cmcm_checkpoint_path}")
            load_model = torch.load(cmcm_checkpoint_path, map_location="cpu")

            cmcm_state_dict = load_model["state_dict"] if "state_dict" in load_model else load_model
            new_state_dict = {}
            for k, v in cmcm_state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            cmcm_state_dict = new_state_dict
            
            cmcm_state_dict.pop("animatediff_config", "")
            missing, unexpected = pipeline.unet.load_state_dict(cmcm_state_dict, strict=False)
            assert len(unexpected) == 0
        
        pipeline = pipeline.to("cuda")

        if omcm_checkpoint_path != "" and os.path.exists(omcm_checkpoint_path):

            name_part = omcm_checkpoint_path.split('/')
            savedir = savedir + f"_omcm_{name_part[-3].split('_')[0]}"

            omcm = Adapter(**model_config.omcm_config.params)

            load_model = torch.load(omcm_checkpoint_path, map_location="cpu")
            # savedir = savedir + f"global_step{load_model['global_step']}_T{model_config.get('omcm_min_step', 700)}"

            omcm_state_dict = load_model['omcm_state_dict']
            new_state_dict = {}
            for k, v in omcm_state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            omcm_state_dict = new_state_dict

            m, u = omcm.load_state_dict(omcm_state_dict, strict=True)
            assert len(u) == 0

            idx = 0
            for _name, _module in unet.down_blocks.named_modules():
                if _module.__class__.__name__ == "CrossAttnDownBlock3D":
                    bound_moudule = Adapted_CrossAttnDownBlock3D_forward.__get__(_module, _module.__class__)
                    setattr(_module, "forward", bound_moudule)
                    setattr(_module, "traj_fea_idx", idx)
                    idx += 1

                elif _module.__class__.__name__ == "DownBlock3D":
                    bound_moudule = Adapted_DownBlock3D_forward.__get__(_module, _module.__class__)
                    setattr(_module, "forward", bound_moudule)
                    setattr(_module, "traj_fea_idx", idx)
                    idx += 1

            omcm = omcm.to(pipeline.device)

            if use_optical_flow:
                print(f'!!!!! dense optical flow !!!!!')
                opt_model = instantiate_from_config(optical_flow_config)
                assert os.path.exists(optical_flow_config.pretrained_model)
                print(f"Loading pretrained motion stage model from {optical_flow_config.pretrained_model}")
                opt_model.load_state_dict(torch.load(optical_flow_config.pretrained_model)['model'])
                opt_model.eval()
                for param in opt_model.parameters():
                    param.requires_grad = False
                num_reg_refine = optical_flow_config.num_reg_refine
                opt_model = opt_model.to(pipeline.device)

        os.makedirs(savedir, exist_ok=True)

        prompts      = model_config.prompt
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
        eval_listdir = [x for x in os.listdir(args.eval_datadir)]
        filtered_eval_listdir = eval_listdir[750:]
        
        for idx, listdir in tqdm(enumerate(filtered_eval_listdir)):
            filedir = '{}/{}'.format(args.eval_datadir, listdir)
            eval_file = [x for x in os.listdir(filedir)]

            text_prompt_file = '{}/text.txt'.format(filedir)
            with open(text_prompt_file, 'r') as f:
                caption = f.readlines()[0]
            
            video_file = '{}/source_video.mp4'.format(filedir)
            cap = cv2.VideoCapture(video_file)
            ret, frame = cap.read()
            original_pose_height = frame.shape[0]
            original_pose_width = frame.shape[1]
            
            # Target pose1
            print('Loading Target Pose 1 K, R, t matrix')
            target_pose1 = '{}/target_poses1.txt'.format(filedir)
            with open(target_pose1, 'r') as f:
                poses = f.readlines()
            poses = [pose.strip().split(' ') for pose in poses[1:]]

            cam_params = [[float(x) for x in pose] for pose in poses]
            cam_params = [Camera(cam_param) for cam_param in cam_params]
            sample_wh_ratio = args.image_width / args.image_height
            pose_wh_ratio = args.original_pose_width / args.original_pose_height
            if pose_wh_ratio > sample_wh_ratio:
                resized_ori_w = args.image_height * pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / args.image_width
            else:
                resized_ori_h = args.image_width / pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / args.image_height
            intrinsic = np.asarray([[cam_param.fx * args.image_width,
                                    cam_param.fy * args.image_height,
                                    cam_param.cx * args.image_width,
                                    cam_param.cy * args.image_height]
                                    for cam_param in cam_params], dtype=np.float32)

            K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]

            c2ws = []
            for cam_pa in cam_params:
                c2ws.append(torch.as_tensor(cam_pa.c2w_mat).unsqueeze(0))
            c2ws = torch.cat(c2ws)
            RT = c2ws[:,:3].reshape(1, 16, -1)
            if model_config.guidance_scale > 1.0:
                RT = torch.cat([torch.zeros_like(RT), RT], dim=0) 
            RT = RT.to(pipeline.device)
            prompt = caption
            n_prompt = None
            traj_features = None
                    
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,

                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                RT = RT,
                traj_features = traj_features,
                omcm_min_step = model_config.get("omcm_min_step", 700),
            ).videos
            samples.append(sample)
            save_name = "_".join(caption.split(" ")) + '_pose1_'
            save_name = save_name.replace(',', '')
            save_videos_grid(sample, f"{video_pth}/{listdir}_{save_name}.mp4")
            save_videos_jpg(sample, f"{image_pth}", f"{listdir}_{save_name}")



            # Target pose 2
            print('Loading Target Pose 2 K, R, t matrix')
            target_pose1 = '{}/target_poses2.txt'.format(filedir)
            with open(target_pose1, 'r') as f:
                poses = f.readlines()
            poses = [pose.strip().split(' ') for pose in poses[1:]]

            cam_params = [[float(x) for x in pose] for pose in poses]
            cam_params = [Camera(cam_param) for cam_param in cam_params]
            sample_wh_ratio = args.image_width / args.image_height
            pose_wh_ratio = args.original_pose_width / args.original_pose_height
            if pose_wh_ratio > sample_wh_ratio:
                resized_ori_w = args.image_height * pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / args.image_width
            else:
                resized_ori_h = args.image_width / pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / args.image_height
            intrinsic = np.asarray([[cam_param.fx * args.image_width,
                                    cam_param.fy * args.image_height,
                                    cam_param.cx * args.image_width,
                                    cam_param.cy * args.image_height]
                                    for cam_param in cam_params], dtype=np.float32)

            K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]

            c2ws = []
            for cam_pa in cam_params:
                c2ws.append(torch.as_tensor(cam_pa.c2w_mat).unsqueeze(0))
            c2ws = torch.cat(c2ws)
            RT = c2ws[:,:3].reshape(1, 16, -1)
            if model_config.guidance_scale > 1.0:
                RT = torch.cat([torch.zeros_like(RT), RT], dim=0) 
            RT = RT.to(pipeline.device)
            prompt = caption
            n_prompt = None
            traj_features = None
                    
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,

                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                RT = RT,
                traj_features = traj_features,
                omcm_min_step = model_config.get("omcm_min_step", 700),
            ).videos
            samples.append(sample)
            save_name = "_".join(caption.split(" ")) + '_pose2_'
            save_name = save_name.replace(',', '')
            save_videos_grid(sample, f"{video_pth}/{listdir}_{save_name}.mp4")
            save_videos_jpg(sample, f"{image_pth}", f"{listdir}_{save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v3.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")
    
    parser.add_argument("--eval_datadir", type=str)
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--original_pose_width", type=int, default=1280, help='the width of the video used to extract camera trajectory')
    parser.add_argument("--original_pose_height", type=int, default=720, help='the height of the video used to extract camera trajectory')

    args = parser.parse_args()
    main(args)
