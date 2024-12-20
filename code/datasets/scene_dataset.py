import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import json
from PIL import Image
import cv2 as cv
import math
import glob



def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 white_bkgd=False
                 ):

        self.instance_dir = os.path.join('../data', data_dir)
        #print(f"data_dir:{data_dir}")
        #print(f"self.instance_dir:{self.instance_dir}")
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        if os.path.exists(os.path.join(self.instance_dir,  "transforms_train.json")):
            print("find transforms_train.json, assuming DNeRF dataset!")
            self.read_DNeRF_dataset(data_dir, img_res, scan_id, white_bkgd)
        elif os.path.exists(os.path.join(self.instance_dir, "cameras.npz")):
            print("find cameras.npz, assuming DTU dataset!")
            self.read_DTU_dataset(data_dir, img_res, scan_id, white_bkgd)
        else:
            print("cannot recognize scene type!")
            raise Exception


    def read_DNeRF_dataset(self,
                            data_dir,
                            img_res,
                            scan_id=0,
                            white_bkgd=False):
        
        contents = json.load(open(os.path.join(self.instance_dir,  "transforms_train.json"))) #  transforms_test  transforms_train
        frames = contents["frames"]
        extension = '.png'
        image = Image.open(os.path.join(self.instance_dir, frames[0]["file_path"] + extension))
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
        
        self.rgb_images = []
        self.intrinsics_all = []
        self.pose_all = []
        self.images_lis = []
        self.time_all = []
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(self.instance_dir, frame["file_path"] + extension)
            #print(f"cam_name be:{cam_name}")
            c2w = np.array(frame["transform_matrix"]) @ blender2opencv # opencv c2w
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(c2w).float())
            
            #image_path = os.path.join(self.instance_dir, cam_name)
            image_path = cam_name
            self.images_lis.append(image_path)

            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_bkgd else np.array([0,0,0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            img = (arr * 255.0).astype(np.uint8)
            # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            # cv.imshow('Window Name',img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            img = img / 255.0
            rgb = img.reshape(-1, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
            time = np.array(frame["time"])
            self.time_all.append(torch.from_numpy(time).float())

            self.n_images = len(self.rgb_images)
            #print("Loaded %d images" % self.n_images)
            self.debug = False

    def read_DTU_dataset(self,
                         data_dir,
                         img_res,
                         scan_id=0,
                         white_bkgd=False):
        self.scan_list = sorted(glob.glob(os.path.join(self.instance_dir, "scan*")))
        self.n_images = len(self.scan_list)
        extension = ".jpg"
        self.n_cams = len(glob.glob( os.path.join(self.scan_list[0],"image", f"*{extension}") ))
        if self.n_cams == 0:
            extension = ".png"
            self.n_cams = len(glob.glob( os.path.join(self.scan_list[0], f"*{extension}") ))
        if self.n_cams == 0:
            print(f"no imgs found in:{os.path.join(self.scan_list[0], f'*{extension}')}")
            raise Exception
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_cams)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_cams)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        #expand intrinsics & pose to match the length of frames
        while len(self.intrinsics_all) < self.n_images:
            N = len(self.intrinsics_all)
            for i in range(N):
                self.intrinsics_all.append(self.intrinsics_all[i])
                self.pose_all.append(self.pose_all[i])

        self.rgb_images = []
        for scan_id in range(len(self.scan_list)):
            scan_path = self.scan_list[scan_id]
            imgs = sorted(glob.glob(os.path.join(scan_path, "image", f"*{extension}")))
            img_path = imgs[scan_id % self.n_cams]
            rgb = rend_util.load_rgb(img_path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.time_all = torch.tensor([i for i in range(self.n_images)]).float()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "time": self.time_all[idx],
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
