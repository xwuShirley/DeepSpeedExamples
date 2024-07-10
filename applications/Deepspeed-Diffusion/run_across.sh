
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
export PATH=$PATH:/home/xiaoxiawu/.local/bin/

cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker0.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out0.log


ssh worker-1
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker1.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out1.log
sleep 1s
exit

ssh worker-2
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker2.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out2.log
sleep 1s
exit

ssh worker-3
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker3.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out3.log
sleep 1s
exit

ssh worker-9
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker4.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out4.log
sleep 1s
exit

ssh worker-8
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker5.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out5.log
sleep 1s
exit

ssh worker-7
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker6.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out6.log
sleep 1s
exit

ssh worker-6
export PATH=$PATH:/home/xiaoxiawu/.local/bin/
cd /vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3dplus-SD21-unclip/wonder3dplus-SD21-unclip
accelerate launch --config_file accelerate_multiple_worker/main_worker7.yaml train_mvdiffusion_image_sd21_unclip.py --config configs/train/ortho/mvdiffusion-mix-unclip-ortho-4views-lvis-res768.yaml&>out7.log
sleep 1s
exit
