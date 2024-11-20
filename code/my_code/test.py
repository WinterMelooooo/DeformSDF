import numpy as np

src = "/home/yktang/DeformSDF/data/DTU/scan24/cameras.npz"
file = np.load(src)
print(type(file["world_mat_0"]))
