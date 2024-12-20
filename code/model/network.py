import torch.nn as nn
import numpy as np
import torch
from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
from model.temporalnerf import DirectTemporalNeRF
from model.transformers import PntTransformer
from pytorch3d.ops import ball_query
from .nerf import Embedding, NeRF
import utils.general as utils

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            geometric_init=True,
            bias=1.0,
            multires_view=0,
            encoding_density = True,
            encoding_var = True,
            encoding_smoothed_pos = True,
            encoding_smoothed_dir = True,
            exclude_ray = True,
            same_smooth_factor = False,
            fix_radius = True,
            particle_radius = 1.0,
            search_raduis_scale = 9.0,
            N_neighbor = 20
    ):
        super().__init__()

        self.raduis = search_raduis_scale * particle_radius
        self.fix_radius = fix_radius
        self.num_neighbor = N_neighbor
        self.encoding_density = encoding_density
        self.encoding_var = encoding_var
        self.encoding_smoothed_pos = encoding_smoothed_pos
        self.encoding_smoothed_dir = encoding_smoothed_dir 
        self.exclude_ray = exclude_ray
        self.same_smooth_factor = same_smooth_factor
        self.particle_radius = particle_radius 
        self.search_raduis_scale = search_raduis_scale 

        # build network
        self.embedding_xyz = Embedding(3, 10)
        in_channels_xyz = self.embedding_xyz.out_channels
        self.embedding_dir = Embedding(3, 4)
        in_channels_dir = self.embedding_dir.out_channels
        if encoding_density:
            self.embedding_density = Embedding(1, 4)
            in_channels_xyz += self.embedding_density.out_channels
        if encoding_var:
            in_channels_xyz += self.embedding_xyz.out_channels
        if encoding_smoothed_pos:
            in_channels_xyz += self.embedding_xyz.out_channels
        if encoding_smoothed_dir:
            in_channels_dir += self.embedding_dir.out_channels

        self.mode = mode
        dims = [d_in + feature_vector_size + in_channels_xyz + in_channels_dir] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires_view > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def update_raduis(self, particle_raduis):
        self.raduis = self.search_raduis_scale * particle_raduis

    def get_particles_direction(self, particles, ro):
        #print(f"particles.shape be:{particles.shape}")
        #print(f"ro.shape be:{ro.shape}")
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs

    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        #print(f"ray_pos.shape be:{ray_pos.shape}")
        #print(f"nn_poses.shape be:{nn_poses.shape}")
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.same_smooth_factor:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)

    def search(self, ray_particles, particles, fix_radius):
        if particles.shape[0] == 0:
            #print("Warning: phys_particles is empty!")
            dists = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, device=ray_particles.device)
            indices = torch.full(
                                    (ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor),  # size 参数是元组
                                    -1,  # 填充值
                                    device=ray_particles.device,
                                    dtype=torch.int64
                                )
            neighbors = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, ray_particles.shape[2], device=ray_particles.device)
            return dists, indices, neighbors, self.raduis
        #print(f"ray_particles.shape be:{ray_particles.shape}")
        #print(f"phys_particles.shape be:{particles.shape}")
        raw_data = particles.repeat(ray_particles.shape[0], 1, 1)
        #print(f"raw_data.shape be:{raw_data.shape}")
        if fix_radius:
            radiis = self.raduis
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radiis, K=self.num_neighbor)
        # else:
        #     radiis = self.get_search_raduis(self.raduis, ray_particles[:,:,-1] - ro[-1], focal)
        #     dists, indices, neighbors = self._ball_query(ray_particles, raw_data, radiis, self.num_neighbor)
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
        #print(f"ray_particles.shape be:{ray_particles.shape}")
        #print(f"rays.shape be:{rays.shape}")
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
        #print(f"neighbors.shape be:{neighbors.shape}")
        #print(f"ray_particles.shape be:{ray_particles.shape}")
        smoothed_pos, density = self.smoothing_position(ray_particles.reshape(-1,3).unsqueeze(0), neighbors, radius, num_nn, exclude_ray=self.exclude_ray)
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

    def forward(self, points, normals, view_dirs, feature_vectors, phys_points, ray_dirs, cam_loc ):
        #print(f"points.shape be:{points.shape}")
        #print(f"normals.shape be:{normals.shape}")
        #print(f"view_dirs.shape be:{view_dirs.shape}")
        #print(f"feature_vectors.shape be:{feature_vectors.shape}")
        #print(f"phys_points.shape be:{phys_points.shape}")
        #print(f"ray_dirs.shape be:{ray_dirs.shape}")
        #print(f"cam_loc.shape be:{cam_loc.shape}")
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        points = points.unsqueeze(0)
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(points, phys_points, self.fix_radius)
        points = points.reshape(ray_dirs.shape[0], -1, 3)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, points, ray_dirs, cam_loc)
        #print(f"pos_like_feats_0: {len(pos_like_feats_0)}, {pos_like_feats_0[0].shape}, {pos_like_feats_0[1].shape}, {pos_like_feats_0[2].shape}, {pos_like_feats_0[3].shape}")
        #print(f"dirs_like_feats_0: {len(dirs_like_feats_0)}, {dirs_like_feats_0[0].shape}, {dirs_like_feats_0[1].shape}")   
        input_feats_0 = torch.cat(pos_like_feats_0+dirs_like_feats_0, dim=1)
        #print(f"rendering input: {rendering_input.shape}")
        #print(f"input feats: {input_feats_0.shape}")
        rendering_input = torch.cat((rendering_input, input_feats_0), dim = 1)
        #print(f"rendering input: {rendering_input.shape}")
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        #print(f"forward over for RenderingNetwork!")
        return x

class VolSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        
        
        multires = 10
        i_embed = 0
        multires_views = 4
        self.embed_fn, input_ch = get_embedder(multires, 3, i_embed)
        self.embedtime_fn, input_ch_time = get_embedder(multires, 1, i_embed)
        # self.embeddirs_fn, input_ch_views = get_embedder(multires_views, 3, i_embed)
        output_ch = 4
        skips = [4]
        self.deform_net = DirectTemporalNeRF(input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=0, input_ch_time=input_ch_time,
                 use_viewdirs=True, embed_fn=self.embed_fn,
                 zero_canonical=True).to('cuda')
        self.netchunk = 1024*64

        conf_transformer = conf.get_config('transformers')
        self.transformers = PntTransformer(conf_transformer)
        if torch.cuda.is_available():
            self.transformers.cuda()

    def forward(self, input, pnt_cloud):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        time = input["time"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        ### add dynamic part
        
        embedded = self.embed_fn(points_flat) # 100352, 63
        # embd_time_discr
        input_frame_time_flat = time[:, None].expand(embedded.shape[0], 1)
        embedded_time = self.embedtime_fn(input_frame_time_flat)
        embedded_times = [embedded_time, embedded_time]
        # embed views
        # embedded_dirs = self.embeddirs_fn(dirs_flat) # 100352, 27
        # embedded = torch.cat([embedded, embedded_dirs], -1) # 100352, 90
        # compute delta
        position_delta_flat = batchify(self.deform_net, self.netchunk)(embedded, embedded_times)
        points_flat = points_flat + position_delta_flat
        points = points_flat.reshape(-1, N_samples, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        res = []
        #print(f"before split, pnt_cloud.shape be:{pnt_cloud.shape}")
        split = utils.split_pnt_cloud(pnt_cloud)
        for pnt_cloud in split:
            res.append(self.transformers(pnt_cloud, time))
        pnt_cloud = utils.merge_pnt_cloud(res)
        #print(f"after merge, pnt_cloud.shape be:{pnt_cloud.shape}")
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, pnt_cloud, ray_dirs, cam_loc[0])
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb_values': rgb_values,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta

        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

            output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights

    def plot_3d(self, querypoints):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(querypoints[:,0], querypoints[:,1], querypoints[:,2], s=1, c='b', marker='.', alpha=0.1)
        ax.legend()
        matplotlib.use('TkAgg')
        plt.show()
        plt.savefig('test.png')
        plt.close()
    
    def get_sdf(self, points_flat, time):
        ### add dynamic part
        #print(f"In get_sdf, points_flat.shape be:{points_flat.shape}")
        #print(f"In get_sdf, time be:{time}")
        embedded = self.embed_fn(points_flat) # 100352, 63
        # embd_time_discr
        input_frame_time_flat = time[:, None].expand(embedded.shape[0], 1)
        embedded_time = self.embedtime_fn(input_frame_time_flat)
        embedded_times = [embedded_time, embedded_time]
        # embed views
        # embedded_dirs = self.embeddirs_fn(dirs_flat) # 100352, 27
        # embedded = torch.cat([embedded, embedded_dirs], -1) # 100352, 90
        # compute delta
        position_delta_flat = batchify(self.deform_net, self.netchunk)(embedded, embedded_times)
        points_flat = points_flat + position_delta_flat
        

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)
        return sdf

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_pos, inputs_time):
        num_batches = inputs_pos.shape[0]

        # out_list = []
        # dx_list = []
        # for i in range(0, num_batches, chunk):
        #     out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
        #     out_list += [out]
        #     dx_list += [dx]
        # return torch.cat(out_list, 0), torch.cat(dx_list, 0)
        
        dx_list = []
        for i in range(0, num_batches, chunk):
            dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            dx_list += [dx]
        return torch.cat(dx_list, 0)
    return ret

