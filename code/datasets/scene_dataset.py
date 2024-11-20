import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import random

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_ids,
                 num_max_frame
                 ):

        self.instance_dir = os.path.join('../data', data_dir)
        self.num_max_frame = num_max_frame
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.n_frames = len(scan_ids)
        scan_ids.sort()
        self.scan_ids = scan_ids
        self.debug = False

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        self.rgb_images = []
        for i in range(len(scan_ids)):

            image_dir = os.path.join(self.instance_dir, f"scan{scan_ids[i]}", "image")
            image_paths = sorted(utils.glob_imgs(image_dir))
            self.n_images = len(image_paths)
            frame_images = []

            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                frame_images.append(torch.from_numpy(rgb).float())
            self.rgb_images.append(frame_images)
            

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        #print(os.path.abspath(self.cam_file))
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        '''
        idx: the idx of a specific camera
        '''
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        frame_start = random.randint(self.scan_ids[0], self.scan_ids[ len(self.scan_ids)-1 ] - self.num_max_frame + 1) if len(self.scan_ids) >= self.num_max_frame else self.scan_ids[0]
        frame_end = min( frame_start+self.num_max_frame-1, self.scan_ids[ len(self.scan_ids)-1 ] )

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
        }

        ground_truth = {
            "rgbs": [ self.rgb_images[ self.scan_ids[frame_idx] ][idx] for frame_idx in range(self.n_frames) ]
        }

        if self.sampling_idx is not None:
            ground_truth["rgbs"] = [ self.rgb_images[ self.scan_ids[frame_idx] ][idx][self.sampling_idx, :] for frame_idx in range(self.n_frames) ]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth,(frame_start, frame_end)

    def collate_fn(self, batch_list):
        if self.debug:
            print(f"the input batch list is like this:\n{batch_list}\n")
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if self.debug:
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
