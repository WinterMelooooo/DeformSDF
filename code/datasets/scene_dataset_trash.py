


import os
import torch
import numpy as np
from .load_blender import load_blender_data
import utils.general as utils
from utils import rend_util
import random

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res
                 ):
        '''
        self.instance_dir = os.path.join('../data', data_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.debug = False

        image_dir = '{0}/train'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_train = []
        self.pose_train = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_train.append(torch.from_numpy(intrinsics).float())
            self.pose_train.append(torch.from_numpy(pose).float())

        self.train_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb[:3,:,:]
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.train_images.append(torch.from_numpy(rgb).float())


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = os.path.join('../data', data_dir)
        images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(data_dir)
        i_train, i_val, i_test = i_split
        self.i_train = i_train
        self.i_val = i_val
        self.i_test = i_test
        self.train_times = times[i_train]
        self.img_res = img_res
        self.n_images_train = len(i_train)
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = os.path.join('../data', data_dir)
        images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(data_dir)
        i_train, i_val, i_test = i_split
        self.i_train = i_train
        self.i_val = i_val
        self.i_test = i_test
        images = images[..., :3]
        images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        times = torch.Tensor(times).to(device)
        reshaped_train_images = []
        train_imgs = images[i_train]
        for i in range(train_imgs.shape[0]):
            reshaped_train_images.append(train_imgs[i].reshape(-1, 3).float())
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K
        intrinsics = torch.from_numpy(intrinsics).float()
        self.train_images = reshaped_train_images
        self.train_times = times[i_train]
        self.near = 2.
        self.far = 6.

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.n_images_train = len(i_train)
        i_train.sort()
        self.debug = False
        self.sampling_idx = None


        self.intrinsics_train = []
        self.pose_train = []
        flip_mat = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ])
        for i in range(len(i_train)):
            img_idx = i_train[i]
            c2w = poses[img_idx].cpu().numpy()
            c2w = flip_mat @ c2w
            w2c = np.linalg.inv(c2w)
            w2c = w2c[:3, :4]
            P = K @ w2c
            P = P[:3, :4]
            intrinsics_SDF_version, pose = rend_util.load_K_Rt_from_P(None, P)
            self.pose_train.append(torch.from_numpy(pose).float())
            self.intrinsics_train.append(torch.from_numpy(intrinsics_SDF_version).float())

        
    def __len__(self):
        return self.n_images_train

    def __getitem__(self, idx):
        '''
        idx: the idx of a specific frame
        '''
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)#(height * width, 2)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_train[idx],
            "pose": self.pose_train[idx],
        }

        ground_truth = {
            "rgb": self.train_images[idx]
        }
        
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.train_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]
        return idx, sample, ground_truth, self.train_times[idx].item()

    def collate_fn(self, batch_list):
        if self.debug:
            print(f"the input batch list is like this:\n{batch_list}\n")
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if self.debug:
                print(f"entry be: {entry}")
                print(f"visiting entry:{entry}")
            if type(entry[0]) is dict:
                if "rgbs" in entry[0].keys():
                    all_parsed.append(entry[0])
                else:
                    # make them all into a new dict
                    ret = {}
                    for k in entry[0].keys():
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    all_parsed.append(ret)
            elif type(entry) is tuple :
                all_parsed.append(entry)
            else:
                all_parsed.append(torch.LongTensor(entry))
        all_parsed[-1] = all_parsed[-1][0]
        if self.debug:
            print(f"return value is like this:\n{tuple(all_parsed)}")
        self.debug = False
        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
