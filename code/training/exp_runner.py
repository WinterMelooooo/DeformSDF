import sys

sys.path.append('../code')
import argparse
import GPUtil

from training.volsdf_train import VolSDFTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--npretrain_epochs', type=int, default=20, help='number of pretrain_epochs to train for')
    parser.add_argument('--ntrain_epochs', type=int, default=2000, help='number of train_epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument('--pretrained_pt_cloud', type=str, default='', help="The cloud point extracted from the pretrained average surface")
    #parser.add_argument('--scene_name',type=str,default="unamed")
    #parser.add_argument('--ckpt', default=None, type=str,
    #                    help="ckpt_path=prefix/TIMESTAMP")
    #parser.add_argument('--temp_vis', default=False, action="store_true")
    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    #print(f"is_continue: {opt.is_continue}")
    #print(f"timestamp: {opt.timestamp}")
    trainrunner = VolSDFTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    npretrain_epochs=opt.npretrain_epochs,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    #scene_name = opt.scene_name,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    scan_id=opt.scan_id,
                                    checkpoint=opt.checkpoint,
                                    do_vis=not opt.cancel_vis,
                                    ntrain_epochs = opt.ntrain_epochs
                                    #IniCkpt = opt.ckpt,
                                    #temp_vis = opt.temp_vis
                                    )

    trainrunner.run()
