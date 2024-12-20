unset LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
# CUDA_VISIBLE_DEVICES=5 python training/exp_runner.py --conf ./confs/dtu_mine.conf --scene_name test --ckpt /home/yktang/VolSDF/exps/dtu_0/2024_10_18_19_10_55
# CUDA_VISIBLE_DEVICES=5 python training/exp_runner.py --conf ./confs/DNeRF.conf --scene_name jumpingjack --is_continue --timestamp 2024_11_22_09_10_38
# CUDA_VISIBLE_DEVICES=6 python training/exp_runner.py --conf ./confs/spring_Gaus.conf --scene_name bun --ckpt /home/yktang/VolSDF/exps/dtu_10/2024_11_07_23_34_02
# CUDA_VISIBLE_DEVICES=3 python training/exp_runner.py --conf ./confs/DNeRF.conf --scene_name jumpingjack --is_continue --timestamp 2024_11_23_15_23_22 --temp_vis
# CUDA_VISIBLE_DEVICES=4 python training/exp_runner.py --conf ./confs/DNeRF_colored_bkgd.conf --scene_name jumpingjack
# CUDA_VISIBLE_DEVICES=3 python training/exp_runner.py --conf ./confs/DNeRF.conf --scene_name jumpingjack 
# CUDA_VISIBLE_DEVICES=4 python training/exp_runner.py --conf ./confs/dnerf.conf --scan_id -1
CUDA_VISIBLE_DEVICES=5 python training/exp_runner.py --conf ./confs/dnerf_transformers_spring_Gaus.conf --scan_id -1
# CUDA_VISIBLE_DEVICES=4 python training/exp_runner.py --conf ./confs/dnerf_transformers.conf --scan_id -1 --is_continue --timestamp 2024_12_20_11_08_00