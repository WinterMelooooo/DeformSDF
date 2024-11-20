'''
import numpy as np
import open3d as o3d
import os

ply_path = r"/home/yktang/VolSDF/surfaces"
'''
'''
file strcture should be like this:
    --ply_path
    ----frame_0
    ------surface_2000.ply
    ----frame_1
    ------surface_2000.ply
    ----frame_2
    ------surface_2000.ply
'''
'''
dest = r"/home/yktang/Deformable3DGS/output/fluid_cut_RealWorldScale_YpZhao_003/FluidPoints/test"
'''
'''
file strcture should be like this:
    --dest
    ----fluid_000.ply
    ----fluid_001.ply
    ----fluid_002.ply
'''
'''

bounding_box = [[-0.3, 0.36], [-1.39, -1.09], [-0.21, 0.2]]
offset = (0,0,0)

def DownSampleAndSave(ply_path, dest:str, bounding_box, offset=(0,0,0)):
    if not os.path.exists(dest):
        os.makedirs(dest)
    # Adjust bounding box by the offset
    bounding_box = [
        [bounding_box[0][0] + offset[0], bounding_box[0][1] + offset[0]],
        [bounding_box[1][0] + offset[1], bounding_box[1][1] + offset[1]],
        [bounding_box[2][0] + offset[2], bounding_box[2][1] + offset[2]]
    ]

    # Load the .ply file
    ply_data = o3d.io.read_point_cloud(ply_path)

    # Convert to numpy array for easier processing
    points = np.asarray(ply_data.points)

    # Extract bounding box limits
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]

    # Define the resolution of the uniform grid
    resolution = 0.01

    # Generate a uniform grid of points within the bounding box
    x_range = np.arange(x_min, x_max, resolution)
    y_range = np.arange(y_min, y_max, resolution)
    z_range = np.arange(z_min, z_max, resolution)

    # Create meshgrid of the bounding box points
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Define a KDTree for the original point cloud to find nearest neighbors
    tree = o3d.geometry.KDTreeFlann(ply_data)

    # Filter points that are inside the fluid space
    fluid_points = []
    for point in all_points:
        # Check if there is a surface point above the current point along the y-axis (negative direction)
        query_point = point.copy()
        query_point[1] += resolution  # Move slightly upwards to ensure we're not on the surface itself
        [k, idx, _] = tree.search_knn_vector_3d(query_point, 1)
        nearest_point = points[idx[0]]
        if nearest_point[1] < point[1]:
            fluid_points.append(point)

    # Convert fluid points to numpy array
    fluid_points = np.array(fluid_points)

    point_cloud = o3d.geometry.PointCloud()

    # 将 numpy 数组的点赋值给 PointCloud 对象
    point_cloud.points = o3d.utility.Vector3dVector(fluid_points)

    # 保存点云为 .ply 文件
    o3d.io.write_point_cloud(dest, point_cloud)

    # Save the downsampled fluid points to an .npz file
    np.savez(dest.replace("ply","npz"), pos=fluid_points)

    print(f"Number of fluid points sampled: {len(fluid_points)}")


ply_files = os.listdir(ply_path)
for ply_file in ply_files:
    ply_file_path = os.path.join(ply_path, ply_file,"surface_2000.ply")
    frame_num = int(ply_file.split("_")[-1])
    dest_path = os.path.join(dest, "fluid_{:03d}.ply".format(frame_num))
    print(f"Processing {ply_file_path} to {dest_path}")
    DownSampleAndSave(ply_file_path, dest_path, bounding_box, offset)
'''
import numpy as np
import open3d as o3d
import os

ply_path = r"/home/yktang/VolSDF/surfaces"
'''
file strcture should be like this:
    --ply_path
    ----frame_0
    ------surface_2000.ply
    ----frame_1
    ------surface_2000.ply
    ----frame_2
    ------surface_2000.ply
'''
dest = r"/home/yktang/VolSDF/test"
'''
file strcture should be like this:
    --dest
    ----fluid_000.ply
    ----fluid_001.ply
    ----fluid_002.ply
'''

bounding_box = [[-0.3, 0.36], [-1.39, -1.09], [-0.21, 0.2]]
offset = (0,0,0)

def DownSampleAndSave(ply_path, dest:str, bounding_box, offset=(0,0,0)):

    # Adjust bounding box by the offset
    bounding_box = [
        [bounding_box[0][0] + offset[0], bounding_box[0][1] + offset[0]],
        [bounding_box[1][0] + offset[1], bounding_box[1][1] + offset[1]],
        [bounding_box[2][0] + offset[2], bounding_box[2][1] + offset[2]]
    ]

    # Load the .ply file
    ply_data = o3d.io.read_point_cloud(ply_path)

    # Convert to numpy array for easier processing
    points = np.asarray(ply_data.points)

    # Extract bounding box limits
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]

    # Define the resolution of the uniform grid
    resolution = 0.01

    # Generate a uniform grid of points within the bounding box
    x_range = np.arange(x_min, x_max, resolution)
    y_range = np.arange(y_min, y_max, resolution)
    z_range = np.arange(z_min, z_max, resolution)

    # Create meshgrid of the bounding box points
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Define a KDTree for the original point cloud to find nearest neighbors
    tree = o3d.geometry.KDTreeFlann(ply_data)

    # Filter points that are inside the fluid space
    fluid_points = []
    for point in all_points:
        # Check if there is a surface point above the current point along the y-axis (negative direction)
        query_point = point.copy()
        query_point[1] += resolution  # Move slightly upwards to ensure we're not on the surface itself
        [k, idx, _] = tree.search_knn_vector_3d(query_point, 1)
        nearest_point = points[idx[0]]
        if nearest_point[1] < point[1]:
            fluid_points.append(point)

    # Convert fluid points to numpy array
    fluid_points = np.array(fluid_points)

    point_cloud = o3d.geometry.PointCloud()

    # 将 numpy 数组的点赋值给 PointCloud 对象
    point_cloud.points = o3d.utility.Vector3dVector(fluid_points)

    # 保存点云为 .ply 文件
    o3d.io.write_point_cloud(dest, point_cloud)

    # Save the downsampled fluid points to an .npz file
    np.savez(dest.replace("ply","npz"), pos=fluid_points)

    print(f"Number of fluid points sampled: {len(fluid_points)}")


ply_files = os.listdir(ply_path)
for ply_file in ply_files:
    ply_file_path = os.path.join(ply_path, ply_file,"surface_2000.ply")
    frame_num = int(ply_file.split("_")[-1])
    dest_path = os.path.join(dest, "fluid_{:03d}_.ply".format(frame_num))
    print(f"Processing {ply_file_path} to {dest_path}")
    DownSampleAndSave(ply_file_path, dest_path, bounding_box, offset)