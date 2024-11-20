from colmap_read_model import read_points3d_binary
import torch
pts = r"/home/yktang/VolSDF/data/DTU_from_4D_20240709/scan65/points3D.bin"
Dest = r"/home/yktang/VolSDF/data/DTU_from_4D_20240709/scan65/points3D.pt"

Lst = []
Dict = read_points3d_binary(pts)
for Point in Dict.values():
    Lst.append(Point.xyz)
ts = torch.tensor(Lst)
torch.save(ts,Dest)
