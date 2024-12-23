import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import trimesh
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from torch.utils.tensorboard import SummaryWriter
from model.transformers import PntTransformer
import glob
import open3d as o3d
import numpy as np

class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.sdf_pretrain_epochs = kwargs['sdf_pretrain_epochs']
        self.neurofluid_pretrain_epochs = kwargs['neurofluid_pretrain_epochs']
        self.ntrain_epochs = kwargs['ntrain_epochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.pretrained_mesh = kwargs['pretrained_mesh']
        self.root_dir_extracted_mesh = None
        self.is_continue = False
        self.cmd = kwargs['cmd']
        self.comment = kwargs['comment']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_string('dataset.scan_id')
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        with open(os.path.join(self.expdir, self.timestamp, 'cmd.txt'), 'w') as f:
            f.write(self.cmd)
        with open(os.path.join(self.expdir, self.timestamp, 'comment.txt'), 'w') as f:
            f.write(self.comment)
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)
        self.plots_dir_sdf = os.path.join(self.plots_dir, 'sdf')
        self.plots_dir_neurofluid = os.path.join(self.plots_dir, 'neurofluid')
        self.plots_dir_train = os.path.join(self.plots_dir, 'train')
        utils.mkdir_ifnotexists(self.plots_dir_sdf)
        utils.mkdir_ifnotexists(self.plots_dir_neurofluid)
        utils.mkdir_ifnotexists(self.plots_dir_train)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        self.tranformers_scheduler_params_subdir = "TransformersSchedulerParameters"
        #self.pnt_renderer_scheduler_params_subdir = "PntRendererSchedulerParameters"

        self.pnt_cloud_path = os.path.join(self.expdir, self.timestamp, 'pnt_cloud')
        utils.mkdir_ifnotexists(self.pnt_cloud_path)

        self.tensorboard_path = os.path.join(self.expdir, self.timestamp, 'tensorboard')
        utils.mkdir_ifnotexists(self.tensorboard_path)

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.tranformers_scheduler_params_subdir))
        #utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.pnt_renderer_scheduler_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam( [{'params': [param for name, param in self.model.named_parameters() if 'transformers' not in name] }], lr=self.lr)
        #self.optimizer = torch.optim.Adam( [{'params': [param for name, param in self.model.named_parameters() if 'transformers' not in name and 'pnt_renderer' not in name] }], lr=self.lr)
        self.optimizer_tranformers = torch.optim.Adam( self.model.transformers.parameters(), lr=self.lr)
        #self.optimizer_pnt_renderer = torch.optim.Adam( self.model.pnt_renderer.parameters(), lr=self.lr)
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = (self.sdf_pretrain_epochs+self.neurofluid_pretrain_epochs+self.ntrain_epochs) * len(self.train_dataset)
        decay_steps_tranformers = self.ntrain_epochs * len(self.train_dataset)
        #decay_steps_pnt_renderer = (self.neurofluid_pretrain_epochs+self.ntrain_epochs) * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))
        self.scheduler_tranformers = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_tranformers, decay_rate ** (1./decay_steps_tranformers))
        #self.scheduler_pnt_renderer = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pnt_renderer, decay_rate ** (1./decay_steps_pnt_renderer))
        self.do_vis = kwargs['do_vis']

        self.sdf_start_epoch = 0
        self.neurofluid_start_epoch = 0
        self.train_start_epoch = 0
        if is_continue:
            self.is_continue = is_continue
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            #print(f"loading from old_checkpnts_dir: {old_checkpnts_dir}")
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.sdf_start_epoch = saved_model_state['pretrain_epoch_sdf']
            self.neurofluid_start_epoch = saved_model_state['pretrain_epoch_neurofluid']
            self.train_start_epoch = saved_model_state['train_epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.tranformers_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler_tranformers.load_state_dict(data["transformers_scheduler_state_dict"])

            #data = torch.load(
            #    os.path.join(old_checkpnts_dir, self.pnt_renderer_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            #self.scheduler_pnt_renderer.load_state_dict(data["pnt_renderer_scheduler_state_dict"])


            self.pretrained_mesh_dir = os.path.join(self.expdir, timestamp, 'plots', 'sdf')
            print(f"pretrained_mesh_dir: {self.pretrained_mesh_dir}")

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)
        self.iteration = 0

    def save_checkpoints(self, pretrain_epoch_sdf=0, pretrain_epoch_neurofluid=0, train_epoch=0):
        pretrain_epoch_neurofluid = self.neurofluid_pretrain_epochs if train_epoch else pretrain_epoch_neurofluid
        pretrain_epoch_sdf = self.sdf_pretrain_epochs if pretrain_epoch_neurofluid+train_epoch else pretrain_epoch_sdf
        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(pretrain_epoch_sdf+pretrain_epoch_neurofluid+train_epoch) + ".pth"))
        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(pretrain_epoch_sdf+pretrain_epoch_neurofluid+train_epoch) + ".pth"))
        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(pretrain_epoch_sdf+pretrain_epoch_neurofluid+train_epoch) + ".pth"))
        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "transformers_scheduler_state_dict": self.scheduler_tranformers.state_dict()},
            os.path.join(self.checkpoints_path, self.tranformers_scheduler_params_subdir, str(pretrain_epoch_sdf+pretrain_epoch_neurofluid+train_epoch) + ".pth"))
        torch.save(
            {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "transformers_scheduler_state_dict": self.scheduler_tranformers.state_dict()},
            os.path.join(self.checkpoints_path, self.tranformers_scheduler_params_subdir, "latest.pth"))

        #torch.save(
        #    {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "pnt_renderer_scheduler_state_dict": self.scheduler_pnt_renderer.state_dict()},
        #    os.path.join(self.checkpoints_path, self.pnt_renderer_scheduler_params_subdir, str(pretrain_epoch_sdf+pretrain_epoch_neurofluid+train_epoch) + ".pth"))
        #torch.save(
        #    {"pretrain_epoch_sdf": pretrain_epoch_sdf, "pretrain_epoch_neurofluid": pretrain_epoch_neurofluid,  "train_epoch": train_epoch, "pnt_renderer_scheduler_state_dict": self.scheduler_pnt_renderer.state_dict()},
        #    os.path.join(self.checkpoints_path, self.pnt_renderer_scheduler_params_subdir, "latest.pth"))

    def run(self):
        print("pretraining sdf...")
        pnt_cloud = torch.empty((1,0,3), device='cuda')
        self.model.change_state(pretrain_sdf_renderer=True)
        for epoch in range(self.sdf_start_epoch, self.sdf_pretrain_epochs):

            if not self.model.pretrain_sdf_renderer or self.pretrained_mesh:
                break
            
            if (epoch+1) % self.checkpoint_freq == 0:
                self.save_checkpoints(pretrain_epoch_sdf=epoch)

            if self.do_vis and epoch % self.plot_freq == 20:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()

                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s,pnt_cloud)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach()}
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

                plt.plot(self.model.implicit_network,
                         indices,
                         plot_data,
                         self.plots_dir_sdf,
                         epoch,
                         self.img_res,
                         self.model,
                         model_input['time'],
                         **self.plot_conf
                         )

                self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                self.iteration += 1
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()

                model_outputs = self.model(model_input, pnt_cloud)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                print(
                    '(sdf){0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, round(loss.item(),2),
                                round(loss_output['rgb_loss'].item(),2),
                                round(loss_output['eikonal_loss'].item(),2),
                                round(psnr.item(),2)))
                self.writer.add_scalars("Losses", {
                        "Total_Loss": loss.item(),
                        "RGB_Loss": loss_output['rgb_loss'].item(),
                        "Eikonal_Loss": loss_output['eikonal_loss'].item()
                    }, self.iteration)
                self.writer.add_scalar("PSNR", psnr.item(), self.iteration)
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()


        if self.pretrained_mesh:
            print("loading pretrained mesh for test...")
            mesh = self.read_surface_mesh(pretrained_mesh_path=self.pretrained_mesh)
            pnt_cloud, estimated_radius = utils.sample_point_cloud_from_surface_mesh(mesh)
            o3d.io.write_point_cloud(os.path.join(self.pnt_cloud_path, 'pnt_cloud.ply'), pnt_cloud)
            
        elif self.is_continue:
            print("continuing from pretrained mesh dir...")
            mesh = self.read_surface_mesh(surface_mesh_path=self.pretrained_mesh_dir)
            pnt_cloud, estimated_radius = utils.sample_point_cloud_from_surface_mesh(mesh)
            o3d.io.write_point_cloud(os.path.join(self.pnt_cloud_path, 'pnt_cloud.ply'), pnt_cloud)
        else:
            print("extracting point cloud...")
            mesh = self.read_surface_mesh(self.root_dir_extracted_mesh)
            pnt_cloud, estimated_radius = utils.sample_point_cloud_from_surface_mesh(mesh)
            o3d.io.write_point_cloud(os.path.join(self.pnt_cloud_path, 'pnt_cloud.ply'), pnt_cloud)
            
        self.model.update_raduis(estimated_radius)
        pnt_cloud = torch.Tensor(np.array(pnt_cloud.points)).cuda().reshape(1,-1,3)
        self.model.change_state(pretrain_pnt_renderer=True)
        print("pretraining neurofluid...")
        for epoch in range(self.neurofluid_start_epoch, self.neurofluid_pretrain_epochs):
            if not self.model.pretrain_pnt_renderer:
                break
            
            if (epoch+1) % self.checkpoint_freq == 0:
                self.save_checkpoints(pretrain_epoch_neurofluid=epoch)

            if self.do_vis and epoch % self.plot_freq == 20:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()

                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s,pnt_cloud)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach()}
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

                plt.plot(self.model.implicit_network,
                         indices,
                         plot_data,
                         self.plots_dir_neurofluid,
                         epoch,
                         self.img_res,
                         self.model,
                         model_input['time'],
                         **self.plot_conf
                         )

                self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                self.iteration += 1
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()

                model_outputs = self.model(model_input, pnt_cloud)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                print(
                    '(nerofluid){0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, round(loss.item(),2),
                                round(loss_output['rgb_loss'].item(),2),
                                round(loss_output['eikonal_loss'].item(),2),
                                round(psnr.item(),2)))
                self.writer.add_scalars("Losses", {
                        "Total_Loss": loss.item(),
                        "RGB_Loss": loss_output['rgb_loss'].item(),
                        "Eikonal_Loss": loss_output['eikonal_loss'].item()
                    }, self.iteration)
                self.writer.add_scalar("PSNR", psnr.item(), self.iteration)
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()
                #self.scheduler_pnt_renderer.step()

        self.model.change_state(e2e=True)
        print("training...")
        for epoch in range(self.train_start_epoch, self.ntrain_epochs):
            if (epoch+1) % self.checkpoint_freq == 0:
                self.save_checkpoints(train_epoch=epoch)
            if self.do_vis and epoch % self.plot_freq == 20:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()
                save_pcd_info = {}
                save_pcd_info['epoch'] = epoch
                save_pcd_info['frame'] = model_input['time'].item()
                save_pcd_info['pnt_cloud_path'] = self.pnt_cloud_path
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s,pnt_cloud, save_pcd_info)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach()}
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

                plt.plot(self.model.implicit_network,
                         indices,
                         plot_data,
                         self.plots_dir_train,
                         epoch,
                         self.img_res,
                         self.model,
                         model_input['time'],
                         **self.plot_conf
                         )
            
            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                self.iteration += 1
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()

                model_outputs = self.model(model_input, pnt_cloud)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                print(
                    '(train){0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, round(loss.item(),2),
                                round(loss_output['rgb_loss'].item(),2),
                                round(loss_output['eikonal_loss'].item(),2),
                                round(psnr.item(),2)))
                self.writer.add_scalars("Losses", {
                        "Total_Loss": loss.item(),
                        "RGB_Loss": loss_output['rgb_loss'].item(),
                        "Eikonal_Loss": loss_output['eikonal_loss'].item()
                    }, self.iteration)
                self.writer.add_scalar("PSNR", psnr.item(), self.iteration)
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()
                #self.scheduler_pnt_renderer.step()
                self.scheduler_tranformers.step()



        self.save_checkpoints(train_epoch=epoch)
        self.writer.close() 

    def get_plot_data(self, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }

        return plot_data


    def read_surface_mesh(self, surface_mesh_path = None, pretrained_mesh_path = None) -> trimesh.Trimesh:
        plot_dir = self.plots_dir_sdf if not surface_mesh_path else surface_mesh_path
        file_pattern = f"{plot_dir}/surface_*_frame_*.ply"
        #print(f"file pattern be: {file_pattern}")
        files = glob.glob(file_pattern)
        file_names = [files[i].split('/')[-1] for i in range(len(files))]
        latest_frame = 0
        latest_file_path = None
        for file_name in file_names:
            frame = int(file_name.split('_')[1])
            if frame > latest_frame:
                latest_frame, latest_file_path = frame, os.path.join(plot_dir, file_name)
        if surface_mesh_path:
            os.system('cp {0} {1}'.format(latest_file_path, self.plots_dir_sdf))
        #print(f"reading extracted mesh: {latest_file_path}")
        latest_file_path = latest_file_path if not pretrained_mesh_path else pretrained_mesh_path
        mesh = trimesh.load(latest_file_path)
        return mesh
    
