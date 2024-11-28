import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from torch.utils.tensorboard import SummaryWriter

class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        #print(kwargs)
        #raise Exception
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.temp_vis = kwargs['temp_vis']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        self.scene_name = kwargs['scene_name']
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
            print(f"timestamp be: {timestamp}")
            is_continue = kwargs['is_continue']
            print(f"is_continue be: {is_continue}")
        
        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.scene_name)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.scene_dir = self.expdir
        self.expdir = os.path.join(self.expdir, self.timestamp)
        utils.mkdir_ifnotexists(self.expdir)        

        # create checkpoints dirs
        self.checkpoints_path = os.path.join( self.expdir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))


        print('Loading data ...')
        self.train_datasets = []
        dataset_conf = self.conf.get_config('dataset')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)
        
        self.n_images_train = self.train_dataset.n_images

        print('Finish loading data. Data-set size: {0}'.format(self.n_images_train))
        '''
        if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
            self.nepochs = int(200000 / self.ds_len)
            print('RUNNING FOR {0}'.format(self.nepochs))
        '''

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
        self.model = utils.get_class(self.conf.get_string('train.model_class')) ( conf=conf_model, deform_conf = self.conf.get_config('deform'), train_frames = self.n_images_train, warmup = self.conf.get_int('train.warmup_epoch') )
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.scene_dir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            self.model.deform.load_weights(old_checkpnts_dir)
        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        #print(f"self.plot_freq be: {self.plot_freq }")
        #raise Exception
        self.writer = SummaryWriter(log_dir=os.path.join(self.expdir,"tensorboard"))
        if kwargs["IniCkpt"]:
            self.load_checkpoint(kwargs["IniCkpt"])
            print(f"loaded ckpt from: {kwargs['IniCkpt']}")

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):
        print("training...")
        iteration = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)
                self.model.deform.save_weights(self.checkpoints_path, epoch)
            #self.do_vis = False
            if (self.do_vis and epoch % self.plot_freq == 0) or self.temp_vis:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                time = model_input['time'].cuda()
                #time = time[0]
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    s['time'] = time
                    out = self.model(s,epoch,exp_dir = self.expdir if self.temp_vis else None)
                    d = {'rgb_values': out['rgb_values'].detach(),
                        'normal_map': out['normal_map'].detach()}
                    res.append(d)
                batch_size = 1
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])
                plot_dir = os.path.join(self.expdir, "plots")

                plt.plot(self.model.implicit_network,
                            self.model.deform,
                        indices,
                        plot_data,
                        plot_dir,
                        epoch,
                        self.img_res,
                        time,
                        **self.plot_conf
                        )

                self.model.train()
                self.temp_vis = False

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                iteration += 1
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['time'] = model_input['time'].cuda()
                model_outputs = self.model(model_input,epoch)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                self.model.deform.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.deform.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                        ground_truth['rgb'].cuda().reshape(-1,3))
                
                '''
                print(
                    f"{self.expname}_{self.timestamp} [{epoch*self.num_cams+data_index}/{self.nepochs*self.num_cams}] ({frame+1}/{self.n_frames}): loss = {loss.item()}, rgb_loss = {loss_output['rgb_loss'].item()}, eikonal_loss = {loss_output['eikonal_loss'].item()}, psnr = {psnr.item()}")
                '''
                self.writer.add_scalars("Losses", {
                    "Total_Loss": loss.item(),
                    "RGB_Loss": loss_output['rgb_loss'].item(),
                    "Eikonal_Loss": loss_output['eikonal_loss'].item()
                }, iteration)
                self.writer.add_scalar("PSNR", psnr.item(), iteration)
                
                
                print(
                    '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                psnr.item()))
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)
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
    
    def load_checkpoint( self, ckpt_path ):
        '''
        ckpt_path=prefix/TIMESTAMP
        '''
        params_path = os.path.join(ckpt_path, "checkpoints", "ModelParameters", "latest.pth")
        self.model.load_state_dict(torch.load(params_path)["model_state_dict"])
