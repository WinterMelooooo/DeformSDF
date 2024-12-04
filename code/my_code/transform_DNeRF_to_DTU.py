import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import os
import torch
import numpy as np


import json
from PIL import Image
import cv2 as cv
import math


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


dataset_dir = "/home/yktang/DeformSDF/data/D-NERF_synthetic_dataset/jumpingjacks"
world_mat_dest = os.path.join(dataset_dir,"cameras_not_normalized.npz")
normalize_dest = os.path.join(dataset_dir,"cameras.npz")
instance_dir = dataset_dir
#print(f"data_dir:{data_dir}")
#print(f"instance_dir:{instance_dir}")


assert os.path.exists(instance_dir), "Data directory is empty"

sampling_idx = None

contents = json.load(open(os.path.join(instance_dir,  "transforms_train.json"))) #  transforms_test  transforms_train
frames = contents["frames"]
extension = '.png'
image = Image.open(os.path.join(instance_dir, frames[0]["file_path"] + extension))
width=image.size[0]
height=image.size[1]
if 'camera_intrinsics' in contents: # real-world scenes # contents_train for ood
    intrinsics = contents['camera_intrinsics'] # contents_train for ood
    cx = intrinsics[0]
    cy = intrinsics[1]
    focal = intrinsics[2]
    intrinsics = np.array([[focal, 0, cx, 0], [0, focal, cy, 0], [0, 0, 1, 0], [0,0,0,1]])
else:
    fovx = float(contents['camera_angle_x']) # contents_train for ood
    focal = fov2focal(fovx, image.size[0])
    cx = width / 2.
    cy = height / 2.
    intrinsics = np.array([[focal, 0, cx, 0], [0, focal, cy, 0], [0, 0, 1, 0], [0,0,0,1]])

rgb_images = []
intrinsics_all = []
pose_all = []
images_lis = []
time_all = []
world_mats = {}
K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
for idx, frame in enumerate(frames):
    cam_name = os.path.join(instance_dir, frame["file_path"] + extension)
    #print(f"cam_name be:{cam_name}")
    c2w = np.array(frame["transform_matrix"]) @ blender2opencv # opencv c2w
    print(f"c2w be: \n{c2w}")
    frame = frames[idx]
    w2c = np.linalg.inv(c2w)
    w2c = w2c[:3, :4]
    P = K @ w2c
    world_mat = np.eye(4)
    world_mat[:3,:4] = P
    world_mats[f"world_mat_{idx}"] = world_mat

np.savez(world_mat_dest, **world_mats)
file = np.load(world_mat_dest)
try:    
    print(file.files()[0])
    print(file[f"{file.files()[0]}"])
except:
    pass

os.system(f"python /home/yktang/DeformSDF/data/preprocess/normalize_cameras.py --input_cameras_file {world_mat_dest} --output_cameras_file {normalize_dest} ")