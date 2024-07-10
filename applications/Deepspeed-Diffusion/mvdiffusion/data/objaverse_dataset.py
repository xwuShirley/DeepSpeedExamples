from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb

def shift_list(lst, n):
    length = len(lst)
    n = n % length  # Ensure n is within the range of the list length
    return lst[-n:] + lst[:-n]

def scale_and_pad_image(image: PIL.Image, scale_factor: float):
    # Open the image
    original_image = image
    # h,w,c = image.shape
    # is_RGBA = True if c==4 else False

    # Get the original size
    original_size = original_image.size
    new_size = [int(original_size[0]*scale_factor), int(original_size[1]*scale_factor)]

    # Scale the image to a smaller size 's'
    scaled_image = original_image.resize(new_size)

    # Create a new image with the original size and a white background

    padded_image = Image.fromarray(np.zeros_like(np.array(original_image)))


    # Calculate the position to paste the scaled image at the center
    paste_position = ((original_size[0] - new_size[0]) // 2, (original_size[1] - new_size[1]) // 2)

    # Paste the scaled image onto the padded image
    padded_image.paste(scaled_image, paste_position)

    return padded_image


class ObjaverseDataset(Dataset):
    def __init__(self,
        root_dir: str,
        num_views: int,
        bg_color: Any,
        img_wh: Tuple[int, int],
        object_list: str,
        groups_num: int=1,
        validation: bool = False,
        random_views: bool = False,
        data_view_num: int = 8,
        num_validation_samples: int = 64,
        num_samples: Optional[int] = None,
        invalid_list: Optional[str] = None,
        trans_norm_system: bool = True,   # if True, transform all normals map into the cam system of front view
        augment_data: bool = False,
        read_normal: bool = True,
        read_color: bool = False,
        read_depth: bool = False,
        read_mask: bool = True,
        mix_color_normal: bool = False,
        random_view_and_domain: bool = False,
        suffix: str = 'png',
        subscene_tag: int = 0,
        random_scale: bool = False,
        data_path_num: int = 1
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        if data_path_num>1:
            self.root_dir = Path('/')
        else:
            self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.bg_color = bg_color
        self.validation = validation
        self.num_samples = num_samples
        self.trans_norm_system = trans_norm_system
        self.augment_data = augment_data
        self.invalid_list = invalid_list
        self.groups_num = groups_num
        print("augment data: ", self.augment_data)
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_color = read_color
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.mix_color_normal = mix_color_normal  # mix load color and normal maps
        self.random_view_and_domain = random_view_and_domain # load normal or rgb of a single view
        self.random_views = random_views
        self.suffix = suffix
        self.subscene_tag = subscene_tag
        self.random_scale = random_scale
        if data_view_num == 6:
            self.view_types  = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        if data_view_num == 8:
            self.view_types  = ['front', 'front_right', 'right', 'back', 'left', 'front_left', 'back_right', 'back_left',]
        elif data_view_num == 4:
            self.view_types  = ['front', 'right', 'back', 'left']
        elif data_view_num == 12:
            self.view_types = ["front", "back", "right", "left", "front_right", "front_left", "back_right", "back_left", "front_right_top", "front_left_top", "back_right_top", "back_left_top", ]
        
        self.normal_text_embeds = torch.load('./mvdiffusion/data/fixed_prompt_embeds_8views/normal_embeds.pt')
        self.color_text_embeds = torch.load('./mvdiffusion/data/fixed_prompt_embeds_8views/clr_embeds.pt')
        self.dict_position1 = {'front':4-1, 'front_left':8-1, 'left':3-1,'back_left':6-1, 'back':2-1,'back_right':5-1,'right':1-1, 'front_right':7-1,}
        
        if data_view_num > 9 :
            self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/thirteen_views"
        else:
            self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"

        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        if object_list is not None:
            if data_path_num>1:
                self.objects = []
                for i_x, x in enumerate(object_list):
                    if 'json' in x:
                        with open(x) as f:
                            tmp = json.load(f)
                        self.objects.extend([f'{root_dir[i_x]}/{el}' for el in tmp]) 
                    else:
                        self.objects.append(x)      
            else:
                with open(object_list) as f:
                    self.objects = json.load(f)
            #self.objects = [os.path.basename(o).replace(".glb", "") for o in self.objects]
        else:
            self.objects = os.listdir(self.root_dir)
            self.objects = sorted(self.objects)

        if self.invalid_list is not None:
            with open(self.invalid_list) as f:
                self.invalid_objects = json.load(f)
            self.invalid_objects = [os.path.basename(o).replace(".glb", "") for o in self.invalid_objects]
        else:
            self.invalid_objects = []
        
        
        #self.all_objects = set(self.objects) - (set(self.invalid_objects) & set(self.objects))
        self.all_objects = self.objects
        print ("how many object?", len(self.all_objects))
        if not validation:
            self.all_objects = self.all_objects[:-num_validation_samples]
        else:
            self.all_objects = self.all_objects[-num_validation_samples:]
        if num_samples is not None:
            self.all_objects = self.all_objects[:num_samples]

        print("loading ", len(self.all_objects), " objects in the dataset")

        if self.mix_color_normal:
            self.backup_data = self.__getitem_mix__(0, self.all_objects[-1])
        else:
            self.backup_data = self.__getitem_norm__(0, self.all_objects[-1])# "66b2134b7e3645b29d7c349645291f78")

    def __len__(self):
        return len(self.objects)*self.total_view
    
    def identify_path(self, root, obj_name1, obj_name2, image_name):
        #print (root, obj_name1, obj_name2, image_name)
        if 'xiaoxiao' in obj_name2:
            
            return os.path.join(root, obj_name1, obj_name2, image_name)
        else:
            image_name1 = self.dict_position1[image_name.split('000_')[-1].split('.png')[0]]
            if 'mask' in image_name:
                return os.path.join(root, obj_name1, obj_name2, f"segmaps_000_{image_name1}.png") 
            else:
                return os.path.join(root, obj_name1, obj_name2, f"{image_name.split('000_')[0]}000_{image_name1}.png") 
        
    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(os.path.join(self.fix_cam_pose_dir,'%03d_%s_RT.txt'%(0, face)))
            poses[face] = RT

        return poses
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_mask(self, img_path, return_type='np', scale_factor=1.0):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        #img = Image.open(img_path).resize(self.img_wh)
        if 'segmaps_' in img_path:
            img = np.max(Image.open(img_path).resize(self.img_wh), axis=2) > 0
        else:
            img = np.array(Image.open(img_path).resize(self.img_wh))
            img = np.float32(img > 0)
        if scale_factor < 1.0:
            img = scale_and_pad_image(img, scale_factor)
        img = np.array(img)

        img = np.float32(img > 0)

        assert len(np.shape(img)) == 2

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img
    
    def load_image(self, img_path, bg_color, alpha, return_type='np', scale_factor=1.0):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = Image.open(img_path).resize(self.img_wh)

        if scale_factor < 1.0:
            img = scale_and_pad_image(img, scale_factor)

        img = np.array(img)
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 3 or img.shape[-1] == 4 # RGB or RGBA

        if img.shape[-1] == 4:
            if alpha is None:
                alpha = img[:, :, 3:] / 255.
            img = img[:, :, :3]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img
    
    def load_depth(self, img_path, bg_color, alpha, return_type='np', scale_factor=1.0):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = Image.open(img_path).resize(self.img_wh)

        if scale_factor < 1.0:
            img = scale_and_pad_image(img, scale_factor)

        img = np.array(img)
        img = img.astype(np.float32) / 65535. # [0, 1]

        img[img > 0.4] = 0
        img = img / 0.4
        
        assert img.ndim == 2 # depth
        img = np.stack([img]*3, axis=-1)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        # print(np.max(img[:, :, 0]))

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img
    
    def load_normal(self, img_path, bg_color, alpha, RT_w2c=None, RT_w2c_cond=None, return_type='np', scale_factor=1.0):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = Image.open(img_path).resize(self.img_wh)

        if scale_factor < 1.0:
            normal = scale_and_pad_image(normal, scale_factor)

        normal = np.array(normal)
        if 'xiaoxiao' not in img_path:
            normal = ((normal / 255.0 * 2 - 1) * 255).astype(np.uint8)
            
        #print (normal)
        assert normal.shape[-1] == 3 or normal.shape[-1] == 4 # RGB or RGBA
        if normal.shape[-1] == 4:
            if alpha is None:
                alpha = img[:, :, 3:] / 255.
            normal = normal[:, :, :3]


        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)

        img = (normal*0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img

    def __len__(self):
        return len(self.all_objects)

    def __getitem_mix__(self, index, debug_object=None):
        if debug_object is not  None:
            object_name =  debug_object #
            set_idx = random.sample(range(0, self.groups_num), 1)[0] # without replacement
        else:
            object_name = self.all_objects[index%len(self.all_objects)]
            set_idx = 0

        if self.augment_data:
            cond_view = random.sample(self.view_types, k=1)[0]
        else:
            cond_view = 'front'

        if random.random() < 0.5:
            read_color, read_normal, read_depth = True, False, False
        else:
            read_color, read_normal, read_depth = False, True, True

        read_normal = read_normal & self.read_normal
        read_depth = read_depth & self.read_depth

        assert (read_color and (read_normal or read_depth)) is False
        
        if not self.random_views:
            view_types = self.view_types
        else:
            view_types = ['front'] + random.sample(self.view_types[1:], k=self.num_views-1)

        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in view_types]

        elevations = []
        azimuths = []

        # get the bg color
        bg_color = self.get_bg_color()
        # random scale imgs
        if self.random_scale:
            resize_scale = random.uniform(0.8, 1.0)
        else:
            resize_scale = 1.0

        if self.read_mask:
            cond_alpha = self.load_mask(self.identify_path(self.root_dir,  object_name[:self.subscene_tag], 
                                                     object_name, "mask_%03d_%s.%s" % (set_idx, cond_view, self.suffix)), 
                                                     return_type='np', scale_factor=resize_scale)
        else:
            cond_alpha = None
        img_tensors_in = [
            self.load_image(self.identify_path(self.root_dir,  object_name[:self.subscene_tag], 
                                         object_name, "rgb_%03d_%s.%s" % (set_idx, cond_view, self.suffix)), 
                                         bg_color, cond_alpha, 
                                         return_type='pt', scale_factor=resize_scale).permute(2, 0, 1)
        ] * self.num_views
        img_tensors_out = []

        text_embeds = []
        for view, tgt_w2c in zip(view_types, tgt_w2cs):
            img_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag], object_name, "rgb_%03d_%s.%s" % (set_idx, view, self.suffix))
            mask_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag], object_name, "mask_%03d_%s.%s" % (set_idx, view, self.suffix))
            normal_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag], object_name, "normals_%03d_%s.%s" % (set_idx, view, self.suffix))
            depth_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag], object_name, "depth_%03d_%s.%s" % (set_idx, view, self.suffix))
            if self.read_mask:
                alpha = self.load_mask(mask_path, return_type='np', scale_factor=resize_scale)
            else:
                alpha = None

            if read_color:                        
                img_tensor = self.load_image(img_path, bg_color, alpha, return_type="pt", scale_factor=resize_scale)
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensors_out.append(img_tensor)
                img_embeds = self.color_text_embeds[view]
                text_embeds.append(img_embeds)

            if read_normal:
                normal_tensor = self.load_normal(normal_path, bg_color, alpha, 
                                                 RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c, 
                                                 return_type="pt", scale_factor=resize_scale).permute(2, 0, 1)
                img_tensors_out.append(normal_tensor)
                normal_embeds = self.normal_text_embeds[view]
                text_embeds.append(normal_embeds)

            if read_depth:
                depth_tensor = self.load_depth(depth_path, bg_color, alpha, 
                                               return_type="pt", scale_factor=resize_scale).permute(2, 0, 1)
                img_tensors_out.append(depth_tensor)
                # depth_prompt = "a depth map of 3D models"
                # text_embeds.append(depth_prompt)

            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)


        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 4 views to train
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)

        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)
        if read_normal or read_depth:
            task_embeddings = normal_task_embeddings
        if read_color:
            task_embeddings = color_task_embeddings
        # print(elevations)
        # print(azimuths)
        
        text_embeds = torch.stack(text_embeds, dim=0)  # [Nv, 77, 1024]

        return {
            # 'elevations_cond': elevations_cond,
            # 'elevations_cond_deg': torch.rad2deg(elevations_cond),
            # 'elevations': elevations,
            # 'azimuths': azimuths,
            # 'elevations_deg': torch.rad2deg(elevations),
            # 'azimuths_deg': torch.rad2deg(azimuths),
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'camera_embeddings': camera_embeddings,
            'task_embeddings': task_embeddings,
            'text_embeds': text_embeds
        }
    

    def __getitem_norm__(self, index, debug_object=None):
        if debug_object is not  None:
            object_name =  debug_object #
            set_idx = random.sample(range(0, self.groups_num), 1)[0] # without replacement
        else:
            object_name = self.all_objects[index%len(self.all_objects)]
            set_idx = 0

        if self.augment_data:
            cond_view = random.sample(self.view_types, k=1)[0]
        else:
            cond_view = 'front'

        if not self.random_views:
            view_types = self.view_types
        else:
            view_types = ['front'] + random.sample(self.view_types[1:], k=self.num_views-1)

        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in view_types]

        elevations = []
        azimuths = []

        # get the bg color
        bg_color = self.get_bg_color()
        # random scale imgs
        if self.random_scale:
            resize_scale = random.uniform(0.6, 1.0)
        else:
            resize_scale = 1.0

        if self.read_mask:
            cond_alpha = self.load_mask(self.identify_path(self.root_dir,  object_name[:self.subscene_tag], 
                                                     object_name, "mask_%03d_%s.%s" % (set_idx, cond_view, self.suffix)), 
                                                     return_type='np', scale_factor=resize_scale)
        else:
            cond_alpha = None
        img_tensors_in = [
            self.load_image(self.identify_path(self.root_dir,  object_name[:self.subscene_tag],
                                          object_name, "rgb_%03d_%s.%s" % (set_idx, cond_view, self.suffix)), 
                                          bg_color, cond_alpha, 
                                          return_type='pt', scale_factor=resize_scale).permute(2, 0, 1)
        ] * self.num_views
        img_tensors_out = []
        normal_tensors_out = []
        text_embeds1 = []
        text_embeds2 = []
        
        for view, tgt_w2c in zip(view_types, tgt_w2cs):
            img_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag], object_name, "rgb_%03d_%s.%s" % (set_idx, view, self.suffix))
            mask_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag], object_name, "mask_%03d_%s.%s" % (set_idx, view, self.suffix))
            if self.read_mask:
                alpha = self.load_mask(mask_path, return_type='np', scale_factor=resize_scale)
            else:
                alpha = None
            if self.read_normal:
                normal_path = self.identify_path(self.root_dir,  object_name[:self.subscene_tag],
                                            object_name, "normals_%03d_%s.%s" % (set_idx, view, self.suffix))
                normal_tensor = self.load_normal(normal_path, bg_color, alpha, 
                                                 RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c, 
                                                 return_type="pt", scale_factor=resize_scale).permute(2, 0, 1)
                normal_tensors_out.append(normal_tensor)
                normal_embeds = self.normal_text_embeds[view]
                text_embeds1.append(normal_embeds)
                
            if self.read_color:                        
                img_tensor = self.load_image(img_path, bg_color, alpha, return_type="pt", scale_factor=resize_scale)
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensors_out.append(img_tensor)
                img_embeds = self.color_text_embeds[view]
                text_embeds2.append(img_embeds)
                    
            # text_embeds = text_embeds1 + text_embeds2
            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        if self.read_color:
            img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        if self.read_normal:
            normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float() # (Nv, 3, H, W)

        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 4 views to train

        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)

        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)
        #text_embeds = torch.stack(text_embeds, dim=0)  # [Nv, 77, 1024]
        return {
            # 'elevations_cond': elevations_cond,
            # 'elevations_cond_deg': torch.rad2deg(elevations_cond),
            # 'elevations': elevations,
            # 'azimuths': azimuths,
            # 'elevations_deg': torch.rad2deg(elevations),
            # 'azimuths_deg': torch.rad2deg(azimuths),
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'normals_out': normal_tensors_out,
            # 'camera_embeddings': camera_embeddings,
            # 'normal_task_embeddings': normal_task_embeddings,
            # 'color_task_embeddings': color_task_embeddings,
            'text_embeds_normals': torch.stack(text_embeds1, dim=0),
            'text_embeds_colors':  torch.stack(text_embeds2, dim=0),
        }

    def __getitem__(self, index):
        try:
            if self.mix_color_normal:
                data = self.__getitem_mix__(index)
            else:
                data = self.__getitem_norm__(index)
            return data

        except:
            try:
                print("error load idx ", index, self.all_objects[index])
                index = np.random.randint(len(self.all_objects)-1)
                return   self.__getitem__(index)
            except:
                print("load error ", self.all_objects[index%len(self.all_objects)] )
                return self.backup_data            


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.num_datasets = len(datasets)

    def __getitem__(self, i):

        chosen = random.choices(self.datasets, self.weights, k=1)[0]
        return chosen[i]

    def __len__(self):
        return max(len(d) for d in self.datasets)

if __name__ == "__main__":
    train_dataset = ObjaverseDataset(
        root_dir="/ghome/l5/xxlong/.objaverse/hf-objaverse-v1/renderings",
        size=(128, 128),
        ext="hdf5",
        default_trans=torch.zeros(3),
        return_paths=False,
        total_view=8,
        validation=False,
        object_list=None,
        views_mode='fourviews'
    )
    data0 = train_dataset[0]
    data1  = train_dataset[50]
    # print(data)
