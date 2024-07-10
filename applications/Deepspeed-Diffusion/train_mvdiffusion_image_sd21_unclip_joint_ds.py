import logging
import warnings
from typing import Callable, List, Optional, Union, Dict, Any

import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPFeatureExtractor, CLIPTokenizer, CLIPTextModel
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.schedulers import KarrasDiffusionSchedulers, PNDMScheduler, DDIMScheduler, DDPMScheduler
from diffusers.utils import deprecate,  randn_tensor
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import transformers
import diffusers

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import deepspeed
from deepspeed.accelerator import get_accelerator

from torchvision.transforms import InterpolationMode
import argparse
from omegaconf import OmegaConf
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.objaverse_dataset import ObjaverseDataset as MVDiffusionDataset
from diffusers.models.vae import DiagonalGaussianDistribution
import os
import numpy as np
from PIL import Image
import math
from tqdm import tqdm
from einops import rearrange, repeat
from torchvision.transforms import InterpolationMode
from einops import rearrange, repeat
from diffusers.schedulers import PNDMScheduler
from collections import defaultdict
from torchvision.utils import make_grid, save_image
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from dataclasses import dataclass
logger = logging.getLogger(__name__)
@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    clip_path: str
    revision: Optional[str]
    train_dataset: Dict
    validation_dataset: Dict
    validation_train_dataset: Dict
    output_dir: str
    seed: Optional[int]
    train_batch_size: int
    validation_batch_size: int
    validation_train_batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    snr_gamma: Optional[float]
    use_8bit_adam: bool
    allow_tf32: bool
    use_ema: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    logging_dir: str
    vis_dir: str
    mixed_precision: Optional[str]
    report_to: Optional[str]
    local_rank: int
    checkpointing_steps: int
    checkpoints_total_limit: Optional[int]
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_sanity_check: bool
    tracker_project_name: str
    data_path: Optional[str]
    trainable_modules: Optional[list]
    use_classifier_free_guidance: bool
    condition_drop_rate: float
    scale_input_latents: bool

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    data_view_num: Optional[int]

    pred_type: str

    drop_type: str
    consistent_decoder: bool

def log_validation(dataloader, vae, feature_extractor, image_encoder, image_normlizer, image_noising_scheduler, tokenizer, text_encoder, 
                   unet, cfg:TrainingConfig, weight_dtype, global_step, name, save_dir):
    logger.info(f"Running {name} ... ")
    
    pipeline = StableUnCLIPImg2ImgPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, image_normalizer=image_normlizer,
        image_noising_scheduler=image_noising_scheduler, tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, unet=unet,
        scheduler=DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )
    pipeline.set_progress_bar_config(disable=True)

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()    

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=get_accelerator().current_device_name()).manual_seed(cfg.seed)
    
    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    for i, batch in enumerate(dataloader):
        # (B, Nv, 3, H, W)
        # if cfg.pred_type == 'color' or cfg.pred_type == 'mix':
        #     imgs_in, imgs_out = batch['imgs_in'], batch['imgs_out']
        # elif cfg.pred_type == 'normal':
        #     imgs_in, imgs_out = batch['imgs_in'], batch['normals_out']
        imgs_in, colors_out, normals_out = batch['imgs_in'], batch['imgs_out'], batch['normals_out']
        
        # text_prompts = batch['text_prompts']
        imgs_in = torch.cat([imgs_in]*2, dim=0)
        imgs_out = torch.cat([normals_out, colors_out], dim=0)
        #prompt_embeddings = batch['text_embeds'] # B, V, L, C
        prompt_embeddings = torch.cat([batch['text_embeds_normals'],  batch['text_embeds_colors']], dim=0) 
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
        # (B*Nv, 3, H, W)
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")

        images_cond.append(imgs_in)
        images_gt.append(imgs_out)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, None, prompt_embeds=prompt_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1,  **cfg.pipe_validation_kwargs
                ).images
                shape = out.shape
                out0, out1 = out[:shape[0]//2], out[shape[0]//2:]
                out = []
                for ii in range(shape[0]//2):
                    out.append(out0[ii])
                    out.append(out1[ii])
                out = torch.stack(out, dim=0)
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
            print ("finish", i)
    images_cond_all = torch.cat(images_cond, dim=0)
    images_gt_all = torch.cat(images_gt, dim=0)
    images_pred_all = {}
    for k, v in images_pred.items():
        images_pred_all[k] = torch.cat(v, dim=0)
    
    nrow = cfg.validation_grid_nrow
    ncol = images_cond_all.shape[0] // nrow
    images_cond_grid = make_grid(images_cond_all, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    images_gt_grid = make_grid(images_gt_all, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    images_pred_grid = {}
    for k, v in images_pred_all.items():
        images_pred_grid[k] = make_grid(v, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    save_image(images_cond_grid, os.path.join(save_dir, f"{global_step}-{name}-cond.jpg"))
    save_image(images_gt_grid, os.path.join(save_dir, f"{global_step}-{name}-gt.jpg"))
    for k, v in images_pred_grid.items():
        save_image(v, os.path.join(save_dir, f"{global_step}-{k}.jpg"))
    torch.cuda.empty_cache()    


def noise_image_embeddings(
                    image_embeds: torch.Tensor,
                    noise_level: int,
                    noise: Optional[torch.FloatTensor] = None,
                    generator: Optional[torch.Generator] = None,
                    image_normalizer: Optional[StableUnCLIPImageNormalizer] = None,
                    image_noising_scheduler: Optional[DDPMScheduler] = None,
                    ):
    """
    Add noise to the image embeddings. The amount of noise is controlled by a `noise_level` input. A higher
    `noise_level` increases the variance in the final un-noised images.

    The noise is applied in two ways
    1. A noise schedule is applied directly to the embeddings
    2. A vector of sinusoidal time embeddings are appended to the output.

    In both cases, the amount of noise is controlled by the same `noise_level`.

    The embeddings are normalized before the noise is applied and un-normalized after the noise is applied.
    """
    if noise is None:
        noise = randn_tensor(
            image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
        )
    noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

    image_embeds = image_normalizer.scale(image_embeds)

    image_embeds = image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

    image_embeds = image_normalizer.unscale(image_embeds)

    noise_level = get_timestep_embedding(
        timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
    )

    # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
    # but we might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    noise_level = noise_level.to(image_embeds.dtype)
    image_embeds = torch.cat((image_embeds, noise_level), 1)
    return image_embeds


def get_ds_config(cfg):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": cfg.train_batch_size*cfg.gradient_accumulation_steps*get_accelerator().device_count(),
        "train_micro_batch_size_per_gpu": cfg.train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "steps_per_print": 2000,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": cfg.mixed_precision == "bf16"},
        "fp16": {
            "enabled": cfg.mixed_precision == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
        "flops_profiler": {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None,
            }
    }
    return ds_config

def is_local_main_process():
    if get_accelerator().device_count() == 1:
        return True

    if get_accelerator().current_device() == 0:
        return True

    return False


def is_main_process():
    if get_accelerator().device_count() == 1:
        return True

    # if deepspeed.comm.get_global_rank() == 0:
    if get_accelerator().current_device() == 0:
        return True

    return False


def dic_to(dic, device):
    for key, value in dic.items():
        dic[key] = dic[key].to(device)
    return dic


def main(cfg: TrainingConfig, args):
    # -------------------------------------------prepare custom log --------------------------------
    vis_dir = os.path.join(cfg.output_dir, cfg.vis_dir)
    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # local_device = get_accelerator().device_name(model_engine.local_rank)
        deepspeed.init_distributed()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        filename=f"{logging_dir}/output.log",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if is_local_main_process():
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        # get_accelerator().manual_seed(cfg.seed)

    # Handle the repository creation
    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))
    ## -------------------------------------- load models -------------------------------- 
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision)
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision)
    image_noising_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_noising_scheduler")
    image_normlizer = StableUnCLIPImageNormalizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_normalizer")
    
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision)
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='text_encoder', revision=cfg.revision)
    # note: official code use PNDMScheduler 
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    if cfg.pretrained_unet_path is None:
        unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    else:
        print("load pre-trained unet from ", cfg.pretrained_unet_path)
        unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_unet_path,  subfolder="unet",  revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
       
    #unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    # unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision)
    if cfg.use_ema:
        # ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        ema_unet = EMAModel(unet.parameters(), model_cls=UNetMV2DConditionModel, model_config=unet.config)
    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # Freeze vae, image_encoder, text_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_normlizer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # count, count1, count2, count3, count4  = 0,0,0,0, 0
    # print ('---vae---')
    # for n, p in vae.named_parameters():
        
    #     count += p.numel()
    #     print (n, p.size(), count)
    # print ('---image_encoder---')
    # for n, p in image_encoder.named_parameters():
    #     count1 += p.numel()
    #     print (n, p.size(), count1)
    # print ('---image_normlizer---')    
    # for n, p in image_normlizer.named_parameters():
    #     count2 += p.numel()
    #     print (n, p.size(), count2)
        
    # print ('---text_encoder---') 
    # for n, p in text_encoder.named_parameters():
    #     count3 += p.numel()
    #     print (n, p.size(), count3)
                                
    # print ('---unet---')
    # for n, p in unet.named_parameters():
    #     count4 += p.numel()
    #     print (n, p.size(), count4)
    # print ('---total---')
    # print (count+count1+count2+count3+count4)
    if cfg.trainable_modules is None:
        unet.requires_grad_(True)
    else:
        unet.requires_grad_(False)
        for name, module in unet.named_modules():
            if name.endswith(tuple(cfg.trainable_modules)):
                for params in module.parameters():
                    params.requires_grad = True     
                               
    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            print("use xformers to speed up")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                if weights: #
                    weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetMV2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(get_accelerator().current_device_name())
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetMV2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        # accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)
    
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True  
        
    # -------------------------------------- optimizer and lr --------------------------------
    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * get_accelerator().device_count()
        )
    # Initialize the optimizer
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params, params_class_embedding = [], []
    for name, param in unet.named_parameters():
        if 'class_embedding' in name:
            params_class_embedding.append(param)
        else:
            params.append(param)
    # optimizer = optimizer_cls(
    #     [
    #         {"params": params, "lr": cfg.learning_rate},
    #         {"params": params_class_embedding, "lr": cfg.learning_rate * cfg.camera_embedding_lr_mult}
    #     ],
    #     betas=(cfg.adam_beta1, cfg.adam_beta2),
    #     weight_decay=cfg.adam_weight_decay,
    #     eps=cfg.adam_epsilon,
    # )
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )
    # -------------------------------------- load dataset --------------------------------
    # Get the training dataset
    train_dataset = MVDiffusionDataset(
        **cfg.train_dataset
    )
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )
    validation_train_dataset = MVDiffusionDataset(
        **cfg.validation_train_dataset
    )

    # if args.local_rank != -1:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=False, drop_last=True)
    #     val_train_sampler = torch.utils.data.distributed.DistributedSampler(validation_train_dataset, shuffle=False, drop_last=True)
    # else:
    train_sampler = None
    val_sampler = None
    val_train_sampler = None

    # DataLoaders creation:
    cfg.dataloader_num_workers=8
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=(train_sampler is None), num_workers=cfg.dataloader_num_workers, sampler=train_sampler
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers, sampler=val_sampler
    )
    validation_train_dataloader = torch.utils.data.DataLoader(
        validation_train_dataset, batch_size=cfg.validation_train_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers, sampler=val_train_sampler
    )

    unet = unet.to(get_accelerator().current_device_name())
    # parameters = filter(lambda p: p.requires_grad, unet.parameters())
    parameters = unet.parameters()
    ds_config = get_ds_config(cfg)
    unet, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args, model=unet, optimizer=optimizer, model_parameters=parameters, lr_scheduler= lr_scheduler, config=ds_config, dist_init_required=True
    )

    if cfg.use_ema:
        ema_unet.to(get_accelerator().current_device_name())
    # -------------------------------------- cast dtype and device --------------------------------
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if cfg.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    image_encoder.to(get_accelerator().current_device_name(), dtype=weight_dtype)
    image_normlizer.to(get_accelerator().current_device_name(), weight_dtype)
    #text_encoder.to(get_accelerator().current_device_name(), dtype=weight_dtype)
    vae.to(get_accelerator().current_device_name(), dtype=weight_dtype)

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(get_accelerator().current_device_name(), dtype=torch.float32)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(get_accelerator().current_device_name(), dtype=torch.float32)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_main_process():
        # tracker_config = dict(vars(cfg))
        tracker_config = {}
        # accelerator.init_trackers(cfg.tracker_project_name, tracker_config)    

    # -------------------------------------- load pipeline --------------------------------
    # pipe = StableUnCLIPImg2ImgPipeline(feature_extractor=feature_extractor,
    #                                     image_encoder=image_encoder,
    #                                     image_normalizer=image_normlizer,
    #                                     image_noising_scheduler= image_noising_scheduler,
    #                                     tokenizer=tokenizer,
    #                                     text_encoder=text_encoder,
    #                                     unet=unet,
    #                                     scheduler=noise_scheduler,
    #                                     vae=vae).to('cuda')
    
    # -------------------------------------- train --------------------------------
    total_batch_size = cfg.train_batch_size * get_accelerator().device_count() * cfg.gradient_accumulation_steps
    generator = torch.Generator(device=get_accelerator().current_device_name()).manual_seed(cfg.seed)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Potentially resume training
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.exists(os.path.join(cfg.output_dir, "checkpoint")):
                path = "checkpoint"
            else:
                dirs = os.listdir(cfg.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
        path = None #'/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip/outputs-512/mvdiffusion-unclip-8views0122real-emberlrx1/checkpoint'
        if path is None:
            print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
        else:
            # ----------------------
            print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(path)#os.path.join(cfg.output_dir, path))
            # global_step = int(path.split("-")[1])
            global_step = 0

            resume_global_step = global_step * cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.gradient_accumulation_steps)        
    # If you want to manually adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.learning_rate
        print ("reset learning rate", param_group['lr'], lr_scheduler.get_last_lr()[0])
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfg.max_train_steps), disable=not is_local_main_process())
    progress_bar.set_description("Steps")
    
    # Main training loop
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfg.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfg.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            # (B, Nv, 3, H, W)
            # if cfg.pred_type in ['color', 'mix']:
            #     imgs_in, imgs_out = batch['imgs_in'], batch['imgs_out']
            # elif cfg.pred_type == 'normal':
            #     imgs_in, imgs_out = batch['imgs_in'], batch['normals_out']

            batch = dic_to(batch, get_accelerator().current_device_name())
            imgs_in, colors_out, normals_out = batch['imgs_in'], batch['imgs_out'], batch['normals_out']
            # repeat  (2B, Nv, 3, H, W)
            imgs_in = torch.cat([imgs_in]*2, dim=0)
            imgs_out = torch.cat([normals_out, colors_out], dim=0)
    
            bnm, Nv = imgs_in.shape[0], imgs_in.shape[1]
            # (B*Nv, 3, H, W)
            imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")
            imgs_in, imgs_out = imgs_in.to(weight_dtype), imgs_out.to(weight_dtype)
            # ------------------------------------encode input text --------------------------------
            # TODO: generate fixed text embedding
            '''
            text_prompt = batch['text_prompt']
            text_prompt_list =  []
            for prompt in text_prompt:
                text_prompt_list += prompt 
            text_inputs = tokenizer(text_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(get_accelerator().current_device_name())
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(text_prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )
            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(get_accelerator().current_device_name())
            else:
                attention_mask = None
            prompt_embeds = text_encoder(text_input_ids.to(get_accelerator().current_device_name()), attention_mask=attention_mask,)
            prompt_embeds = prompt_embeds[0]
            '''
            # (B, Nv, N, C) # text embedding (B, Nv, Nce), including 'human, {normal / color}, {view}'
            prompt_embeddings = torch.cat([batch['text_embeds_normals'],  batch['text_embeds_colors']], dim=0) 
            # prompt_embeddings = batch['text_embeds']
            # (B*Nv, N, C)
            prompt_embeds = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
            prompt_embeds = prompt_embeds.to(weight_dtype)  # BV, L, C
            # ------------------------------------Encoder input image --------------------------------                

            imgs_in_proc = TF.resize(imgs_in, (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC)
            # do the normalization in float32 to preserve precision
            imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std).to(weight_dtype)        
            # (B*Nv, 1024)
            image_embeddings = image_encoder(imgs_in_proc).image_embeds

            noise_level =  torch.tensor([0], device=get_accelerator().current_device_name())
            # (B*Nv, 2048)
            image_embeddings = noise_image_embeddings(image_embeddings, noise_level, generator=generator, image_normalizer=image_normlizer, 
                                                        image_noising_scheduler=image_noising_scheduler).to(weight_dtype)  
            
            #--------------------------------------vae input and output latents ---------------------------------------  
            cond_vae_embeddings = vae.encode(imgs_in * 2.0 - 1.0).latent_dist.mode() # 
            if cfg.scale_input_latents:
                # cond_vae_embeddings = noise_scheduler.scale_mode_input(cond_vae_embeddings) 
                cond_vae_embeddings *=  vae.config.scaling_factor
            
            # sample outputs latent
            latents = vae.encode(imgs_out * 2.0 - 1.0).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # same noise for different views of the same object
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz // cfg.num_views,), device=latents.device)
            timesteps = repeat(timesteps, "b -> (b v)", v=cfg.num_views)
            timesteps = timesteps.long()                

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Conditioning dropout to support classifier-free guidance during inference. For more details
            # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
            if cfg.use_classifier_free_guidance and cfg.condition_drop_rate > 0.:
                if cfg.drop_type == 'drop_as_a_whole':
                    # drop a group of normals and colors as a whole
                    random_p = torch.rand(bnm, device=latents.device, generator=generator)

                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - (
                        (random_p >= cfg.condition_drop_rate).to(image_mask_dtype)
                        * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bnm, 1, 1, 1, 1).repeat(1, Nv, 1, 1, 1)
                    image_mask = rearrange(image_mask, "B Nv C H W -> (B Nv) C H W")
                    # Final image conditioning.
                    #image_mask = torch.cat([image_mask]*2, dim=0)
                    #import pdb; pdb.set_trace()
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - (
                        (random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype)
                    )
                    clip_mask = clip_mask.reshape(bnm, 1,  1).repeat(1, Nv,  1)
                    clip_mask = rearrange(clip_mask, "B Nv C -> (B Nv) C")
                    #clip_mask = torch.cat([clip_mask]*2, dim=0)
                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings
                elif cfg.drop_type == 'drop_independent':
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)

                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - (
                        (random_p >= cfg.condition_drop_rate).to(image_mask_dtype)
                        * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - (
                        (random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype)
                    )
                    clip_mask = clip_mask.reshape(bsz, 1, 1)
                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings
                elif cfg.drop_type == 'drop_joint':
                    # randomly drop all independently
                    random_p = torch.rand(bsz//2, device=latents.device, generator=generator)

                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - (
                        (random_p >= cfg.condition_drop_rate).to(image_mask_dtype)
                        * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype)
                    )
                    #image_mask = torch.cat([image_mask]*2, dim=0)
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - (
                        (random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype)
                    )
                    #clip_mask = torch.cat([clip_mask]*2, dim=0)
                    clip_mask = clip_mask.reshape(bsz, 1, 1)
                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings
            # (B*Nv, 8, Hl, Wl)
            latent_model_input = torch.cat([noisy_latents, cond_vae_embeddings], dim=1)

            model_pred = unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeddings,
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
                # target = noise_scheduler._get_prev_sample(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}") 

            if cfg.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean").to(weight_dtype)
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                # rebalance the sample-wise losses with their respective loss weights.
                # Finally, we take the mean of the rebalanced loss.
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean().to(weight_dtype)                    

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = loss.repeat(get_accelerator().device_count(), cfg.train_batch_size)
            deepspeed.comm.all_gather_into_tensor(avg_loss, loss.repeat(cfg.train_batch_size))
            avg_loss = avg_loss.mean()
            train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

            if is_main_process():
                print(f'step {step}; train_loss {train_loss}')
    
            # Backpropagate
            # print(loss.data)
            # accelerator.backward(loss)

            # if accelerator.sync_gradients and cfg.max_grad_norm is not None:
            #     accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()

            unet.backward(loss)
            if global_step != 0:
                unet.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            if True:
                if cfg.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                # accelerator.log(logs, step=global_step)
                # train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0: # or  global_step == 1:
                    save_path = os.path.join(cfg.output_dir, "checkpoint")
                    unet.save_checkpoint(save_path, global_step)
                    if is_main_process():
                        
                        try:
                            unet.module.save_pretrained(os.path.join(cfg.output_dir, f"unet-{global_step}/unet"))
                        except:
                            unet.save_pretrained(os.path.join(cfg.output_dir, f"unet-{global_step}/unet"))
                        logger.info(f"Saved state to {save_path}")

                if global_step % cfg.validation_steps == 0 or (cfg.validation_sanity_check and global_step == 1):
                    if is_main_process():
                        if cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        torch.cuda.empty_cache()
                        log_validation(
                            validation_dataloader,
                            vae,
                            feature_extractor,
                            image_encoder,
                            image_normlizer,
                            image_noising_scheduler,
                            tokenizer,
                            text_encoder,
                            unet,
                            cfg,
                            weight_dtype,
                            global_step,
                            'validation',
                            vis_dir
                        )
                        # log_validation(
                        #     validation_train_dataloader,
                        #     vae,
                        #     feature_extractor,
                        #     image_encoder,
                        #     image_normlizer,
                        #     image_noising_scheduler,
                        #     tokenizer,
                        #     text_encoder,
                        #     unet,
                        #     cfg,
                        #     weight_dtype,
                        #     global_step,
                        #     'validation_train',
                        #     vis_dir
                        # )                       
                        if cfg.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                          
            if is_main_process(): 
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    # accelerator.wait_for_everyone()

    # if is_main_process():
    #     if cfg.use_ema:
    #         ema_unet.copy_to(unet.parameters())
    #     pipeline = StableUnCLIPImg2ImgPipeline(
    #         image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=unet, safety_checker=None,
    #         scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
    #         **cfg.pipe_kwargs
    #     )            
    #     os.makedirs(os.path.join(cfg.output_dir, "ckpts"), exist_ok=True)
    #     pipeline.save_pretrained(os.path.join(cfg.output_dir, "ckpts"))

    # accelerator.end_training()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1, help="local rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    # override cfg.local_rank
    local_rank = args.local_rank
    if local_rank != -1 and local_rank != cfg.local_rank:
        cfg.local_rank = local_rank

    main(cfg, args)
    
    # device = 'cuda'
    # ## -------------------------------------- load models -------------------------------- 
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision)
    # feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision)
    # image_noising_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_noising_scheduler")
    # image_normlizer = StableUnCLIPImageNormalizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_normalizer")
    
    # tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision)
    # text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='text_encoder', revision=cfg.revision)
    
    # noise_scheduler = PNDMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    # vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    # unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    # # unet = UNetMV2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision,
    #                                                 #   **cfg.unet_from_pretrained_kwargs
    #                                                 #   )
    

    # if cfg.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             print(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #         print("use xformers.")
           
    # # from diffusers import StableUnCLIPImg2ImgPipeline
    # # -------------------------------------- load pipeline --------------------------------
    # pipe = StableUnCLIPImg2ImgPipeline(feature_extractor=feature_extractor,
    #                                     image_encoder=image_encoder,
    #                                     image_normalizer=image_normlizer,
    #                                     image_noising_scheduler= image_noising_scheduler,
    #                                     tokenizer=tokenizer,
    #                                     text_encoder=text_encoder,
    #                                     unet=unet,
    #                                     scheduler=noise_scheduler,
    #                                     vae=vae).to('cuda')
    
    # # -------------------------------------- input --------------------------------
    # # image =  Image.open('test/woman.jpg')
    # # w, h = image.size
    # # image = np.asarray(image)[:w, :w, :]
    # # image_in = Image.fromarray(image).resize((768, 768))
    
    # im_path = '/mnt/pfs/users/longxiaoxiao/data/test_images/syncdreamer_testset/box.png'
    # rgba =  np.array(Image.open(im_path)) / 255.0
    # rgb = rgba[:,:,:3]
    # alpha = rgba[:,:,3:4]
    # bg_color = np.array([1., 1., 1.])
    # image_in = rgb * alpha + (1 - alpha) * bg_color[None,None,:]
    # image_in = Image.fromarray((image_in * 255).astype(np.uint8)).resize((768, 768))
    # res = pipe(image_in, 'a rendering image of 3D models, left view, normal map.').images[0]
    # res.save("unclip.png")