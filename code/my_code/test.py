import numpy as np

src = "/home/yktang/DeformSDF/data/D-NERF_synthetic_dataset/jumpingjacks/cameras_not_normalized.npz"
file = np.load(src)
print(file.files)
