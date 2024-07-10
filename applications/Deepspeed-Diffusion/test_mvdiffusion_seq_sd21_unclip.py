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
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import transformers
import diffusers
import accelerate
from accelerate import Accelerator
from torchvision.transforms import InterpolationMode
import argparse
from omegaconf import OmegaConf
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
# from mvdiffusion.data.objaverse_dataset import ObjaverseDataset as MVDiffusionDataset

from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset
from diffusers.models.vae import DiagonalGaussianDistribution
from accelerate.logging import get_logger
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
logger = get_logger(__name__, log_level="INFO")
@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: str
    clip_path: str
    revision: Optional[str]
    train_dataset: Dict
    validation_dataset: Dict
    validation_train_dataset: Dict
    save_dir: str
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
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool

def log_validation(dataloader, vae, feature_extractor, image_encoder, image_normlizer, image_noising_scheduler, tokenizer, text_encoder, 
                   unet, cfg:TrainingConfig, accelerator, weight_dtype, global_step, name, save_dir):
    logger.info(f"Running {name} ... ")
    
    pipeline = StableUnCLIPImg2ImgPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, image_normalizer=image_normlizer,
        image_noising_scheduler=image_noising_scheduler, tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, unet=accelerator.unwrap_model(unet), 
        scheduler=DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )
    pipeline.set_progress_bar_config(disable=True)

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()    

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
    
    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    for i, batch in enumerate(dataloader):
        # (B, Nv, 3, H, W)
        if cfg.pred_type == 'color' or cfg.pred_type == 'mix':
            imgs_in, imgs_out = batch['imgs_in'], batch['imgs_in']
        elif cfg.pred_type == 'normal':
            imgs_in, imgs_out = batch['imgs_in'], batch['normals_out']
            
       #imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)
        # text_prompts = batch['text_prompts']
        prompt_embeddings = batch['text_embeds'] # B, V, L, C
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        images_cond.append(imgs_in)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, None, prompt_embeds=prompt_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1,  **cfg.pipe_validation_kwargs
                ).images
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
            print ("finish", i)
    images_cond_all = torch.cat(images_cond, dim=0)
    images_pred_all = {}
    for k, v in images_pred.items():
        images_pred_all[k] = torch.cat(v, dim=0)
    
    nrow = cfg.validation_grid_nrow
    ncol = images_cond_all.shape[0] // nrow
    images_cond_grid = make_grid(images_cond_all, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    #images_gt_grid = make_grid(images_gt_all, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    images_pred_grid = {}
    for k, v in images_pred_all.items():
        images_pred_grid[k] = make_grid(v, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    save_image(images_cond_grid, os.path.join(save_dir, f"{global_step}-{name}-cond.jpg"))
    #save_image(images_gt_grid, os.path.join(save_dir, f"{global_step}-{name}-gt.jpg"))
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


def main(cfg: TrainingConfig):
    # -------------------------------------------prepare custom log and accelaeator --------------------------------
    # override local_rank with envvar
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank not in [-1, cfg.local_rank]:
        cfg.local_rank = env_local_rank

    vis_dir = os.path.join(cfg.save_dir, cfg.vis_dir)
    logging_dir = os.path.join(cfg.save_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.save_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
        project_config=accelerator_project_config,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.save_dir, 'config.yaml'))
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
        unet = UNetMV2DConditionModel.from_pretrained(cfg.pretrained_unet_path, subfolder="unet", revision=cfg.revision)

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


    # if cfg.trainable_modules is None:
    #     unet.requires_grad_(True)
    # else:
    #     unet.requires_grad_(False)
    #     for name, module in unet.named_modules():
    #         if name.endswith(tuple(cfg.trainable_modules)):
    #             for params in module.parameters():
    #                 params.requires_grad = True     
                               
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
        def save_model_hook(models, weights, save_dir):
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(save_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                if weights: #
                    weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetMV2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    # if cfg.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True  
        
    # # -------------------------------------- optimizer and lr --------------------------------
    # if cfg.scale_lr:
    #      = (
    #         cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
    #     )
    cfg.learning_rate = 0.01
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
    optimizer = optimizer_cls(
        [
            {"params": params, "lr": cfg.learning_rate},
            {"params": params_class_embedding, "lr": cfg.learning_rate * cfg.camera_embedding_lr_mult}
        ]
    )
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )
    # -------------------------------------- load dataset --------------------------------
    # Get the training dataset
    # train_dataset = MVDiffusionDataset(
    #     **cfg.train_dataset
    # )
    # validation_dataset = MVDiffusionDataset(
    #     **cfg.validation_dataset
    # )
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )

    # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.dataloader_num_workers,
    # )
    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    # )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_train_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )
    # Prepare everything with our `accelerator`.
    unet, optimizer, validation_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, validation_dataloader, lr_scheduler
    )

    # -------------------------------------- cast dtype and device --------------------------------
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    image_normlizer.to(accelerator.device, weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    # clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    # num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     # tracker_config = dict(vars(cfg))
    #     tracker_config = {}
    #     accelerator.init_trackers(cfg.tracker_project_name, tracker_config)    

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
    
    # -------------------------------------- train --------------------------------
    # Potentially resume training
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
        accelerator,
        weight_dtype,
        0,
        'validation',
        vis_dir
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
    
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