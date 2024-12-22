"""
Renderer
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .nerf import Embedding, NeRF
from pytorch3d.ops import ball_query

class RenderNet(nn.Module):
    def __init__(self, cfg):
        super(RenderNet, self).__init__()
        self.search_raduis_scale = cfg.get_int('search_raduis_scale')
        self.raduis = cfg.get_int('particle_radius') * cfg.get_int('search_raduis_scale')
        self.num_neighbor = cfg.get_int('N_neighbor')
        self.fix_radius = cfg.get_bool('fix_radius')
        self.use_mask = cfg.get_bool('use_mask')
        self.encoding_density = cfg.get_bool('encoding_density')
        self.encoding_var = cfg.get_bool('encoding_var')
        self.encoding_smoothed_pos = cfg.get_bool('encoding_smoothed_pos')
        self.encoding_smoothed_dir = cfg.get_bool('encoding_smoothed_dir')
        self.encoding_exclude_ray = cfg.get_bool('encoding_exclude_ray')
        self.encoding_same_smooth_factor = cfg.get_bool('encoding_same_smooth_factor')

        # build network
        self.embedding_xyz = Embedding(3, 10)
        in_channels_xyz = self.embedding_xyz.out_channels
        self.embedding_dir = Embedding(3, 4)
        in_channels_dir = self.embedding_dir.out_channels
        if self.encoding_density:
            self.embedding_density = Embedding(1, 4)
            in_channels_xyz += self.embedding_density.out_channels
        if self.encoding_var:
            in_channels_xyz += self.embedding_xyz.out_channels
        if self.encoding_smoothed_pos:
            in_channels_xyz += self.embedding_xyz.out_channels
        if self.encoding_smoothed_dir:
            in_channels_dir += self.embedding_dir.out_channels
        self.nerf_coarse = NeRF(in_channels_xyz=in_channels_xyz, in_channels_dir=in_channels_dir)
        self.nerf_fine = NeRF(in_channels_xyz=in_channels_xyz, in_channels_dir=in_channels_dir)

    
    def get_particles_direction(self, particles, ro):
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs

    
    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.encoding_same_smooth_factor:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)
    
    
    def search(self, ray_particles, particles, fix_radius):
        ray_particles = ray_particles.reshape(-1, 3).unsqueeze(0)
        raw_data = particles.repeat(ray_particles.shape[0], 1, 1)
        if fix_radius:
            radiis = self.raduis
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radiis, K=self.num_neighbor)
        return dists, indices, neighbors, radiis
    
        
    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles, rays, ro, sigma_only=False):
        """
        pos like feats
            1. smoothed positions
            2. ref hit pos, i.e., ray position
            3. density
            3. variance
        dir like feats
            1. hit direction, i.e., ray direction
            2. main direction after PCA
        """
        # calculate mask
        nn_mask = dists.ne(0)
        num_nn = nn_mask.sum(-1, keepdim=True)

        # hit pos and hit direction (basic in NeRF formulation)
        pos_like_feats = []
        hit_pos = ray_particles.reshape(-1,3)
        hit_pos_embedded = self.embedding_xyz(hit_pos)
        pos_like_feats.append(hit_pos_embedded)
        if not sigma_only:
            hit_dir = rays
            hit_dir_embedded = self.embedding_dir(hit_dir)
            hit_dir_embedded = torch.repeat_interleave(hit_dir_embedded, repeats=ray_particles.shape[1], dim=0)
            dir_like_feats = []
            dir_like_feats.append(hit_dir_embedded)
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles.reshape(-1,3).unsqueeze(0), neighbors, radius, num_nn, exclude_ray=self.encoding_exclude_ray)
        smoothed_dir = self.get_particles_direction(smoothed_pos.reshape(-1, 3), ro)
        # density
        if self.encoding_density:
            density_embedded = self.embedding_density(density.reshape(-1, 1))
            pos_like_feats.append(density_embedded)
        # smoothed pos
        if self.encoding_smoothed_pos:
            smoothed_pos_embedded = self.embedding_xyz(smoothed_pos.reshape(-1, 3))
            pos_like_feats.append(smoothed_pos_embedded)
        # variance
        if self.encoding_var:
            ray_particles = ray_particles.reshape(-1,3).unsqueeze(0)
            vec_pp2rp = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            vec_pp2rp[nn_mask] = (neighbors - ray_particles.unsqueeze(-2))[nn_mask]
            vec_pp2rp_mean = vec_pp2rp.sum(-2) / (num_nn+1e-12)
            variance = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            variance[nn_mask] = ((vec_pp2rp - vec_pp2rp_mean.unsqueeze(-2))**2)[nn_mask]
            variance = variance.sum(-2) / (num_nn+1e-12)
            variance_embedded = self.embedding_xyz(variance.reshape(-1,3))
            pos_like_feats.append(variance_embedded)
        # smoothed dir
        if self.encoding_smoothed_dir:
            smoothed_dir_embedded = self.embedding_dir(smoothed_dir)
            dir_like_feats.append(smoothed_dir_embedded)
        if not sigma_only:
            return pos_like_feats, dir_like_feats, num_nn
        else:
            return pos_like_feats
        
    
    def forward(self, ray_particles_0, physical_particles, ro, rays):
        """
        physical_particles: N_particles, 3
        ray_particles: N_ray, N_samples, 3
        zvals: N_rays, N_samples
        ro: 3, camera location
        rays: N_rays, 6
        """
        N_samples = ray_particles_0.shape[1]
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro)
        input_feats_0 = torch.cat(pos_like_feats_0+dirs_like_feats_0, dim=1)
        # predict rgbsigma
        rgbsigma_0 = self.nerf_coarse(input_feats_0)
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        if self.use_mask:
            rgbsigma_0 = (rgbsigma_0*mask_0).view(-1, N_samples, 4)
        else:
            rgbsigma_0 = rgbsigma_0.view(-1, N_samples, 4)
        return (rgbsigma_0[..., :3]).reshape(-1, 3)

        