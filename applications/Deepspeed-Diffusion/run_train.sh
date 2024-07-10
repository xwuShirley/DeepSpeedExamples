#!/bin/bash --login

export TORCH_EXTENSIONS_DIR=/home/czh5/.cache/polaris_torch_extensions

# accelerate launch --main_process_port 45900 --config_file 1gpu.yaml \
# train_eval_sd21_unclip1.py --config configs/train/ortho/mix-8views-lvis-res768_validate.yaml
# #--config_file 16gpu_ds.yaml

# python -m accelerate.commands.launch  --main_process_port 45900 --config_file 4gpu.yaml \
# train_mvdiffusion_image_sd21_unclip.py \
# --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-8views-lvis-res384.yaml

deepspeed train_mvdiffusion_image_sd21_unclip_ds.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-8views-lvis-res384.yaml --deepspeed

# python -m accelerate.commands.launch  --main_process_port 45900 --config_file 4gpu.yaml \
# train_mvdiffusion_image_sd21_unclip_joint.py --config configs/train/ortho/joint-512.yaml

# python -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None train_mvdiffusion_image_sd21_unclip_joint.py --config configs/train/ortho/joint-512.yaml --deepspeed
# deepspeed train_mvdiffusion_image_sd21_unclip_joint_ds.py --config configs/train/ortho/joint-512.yaml --deepspeed
