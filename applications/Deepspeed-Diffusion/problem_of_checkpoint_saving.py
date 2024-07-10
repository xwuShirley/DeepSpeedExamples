
from accelerate import Accelerator
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPFeatureExtractor, CLIPTokenizer, CLIPTextModel
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.schedulers import KarrasDiffusionSchedulers, PNDMScheduler, DDIMScheduler, DDPMScheduler
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
import os  
from accelerate.utils import ProjectConfiguration

image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
image_noising_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_noising_scheduler")
image_normlizer = StableUnCLIPImageNormalizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_normalizer")

tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', revision=args.revision)
# note: official code use PNDMScheduler 
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNetMV2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, **args.unet_from_pretrained_kwargs)
global_step = 0
logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    project_config=accelerator_project_config,
)
unet= accelerator.prepare(unet )
if accelerator.is_main_process:

    save_path = os.path.join(args.output_dir, "checkpoint")
    accelerator.save_state(save_path)
    try:
        unet.module.save_pretrained(os.path.join(args.output_dir, f"unet-{global_step}/unet"))
    except:
        unet.save_pretrained(os.path.join(args.output_dir, f"unet-{global_step}/unet"))
    print(f"Saved state to {save_path}")
