unset LD_LIBRARY_PATH
# CUDA_VISIBLE_DEVICES=5 python training/exp_runner.py --conf ./confs/dtu_mine.conf --scene_name test --ckpt /home/yktang/VolSDF/exps/dtu_0/2024_10_18_19_10_55
# CUDA_VISIBLE_DEVICES=5 python training/exp_runner.py --conf ./confs/dtu_mine.conf --scene_name test --is_continue --timestamp 2024_11_06_21_20_23
CUDA_VISIBLE_DEVICES=6 python training/exp_runner.py --conf ./confs/spring_Gaus.conf --scene_name bun --ckpt /home/yktang/VolSDF/exps/dtu_10/2024_11_07_23_34_02
