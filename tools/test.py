import numpy as np
from colmap_read_model import read_points3d_binary

file_path = r"D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_XmZhu_Origin_Not_Rotated_2_DTU\sparse\0\points3D.bin"

files = read_points3d_binary(file_path)
print(len(files))
print(min(files.keys()))
print(max(files.keys()))
print(files[9996])