import numpy as np
import open3d as o3d

ply_path = r"/home/yktang/VolSDF/exps/dtu_66/2024_10_08_22_24_48/plots/surface_2000.ply"
# Load the PLY file
pcd = o3d.io.read_point_cloud(ply_path)

# Convert to numpy array
points = np.asarray(pcd.points)

# Save to XYZ file
xyz_path = ply_path.replace('.ply', '.xyz')
np.savetxt(xyz_path, points, delimiter=' ')