import numpy as np
import os

file_path = r"/home/yktang/VolSDF/data/DTU_from_4D_20240709_ypzhao_resized_000/bbs.npz"

file = np.load(file_path)
print(file.files)
for name in file.files:
    print(name)
    print(file[name])
    print("\n")
