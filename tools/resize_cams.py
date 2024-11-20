import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colmap_read_model as read_model  # 你之前提供的脚本处理 .bin 文件
from pathlib import Path

sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_Rotate\sparse'  # 原始sparse文件夹路径
new_sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_Resized_after_Rotate\sparse'  # 新的sparse文件夹路径
npy_file = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_YpZhao_Resized\poses_bounds.npy'  # poses_bounds.npy 文件路径


# 1. 读取 poses_bounds.npy 文件
def load_poses_bounds(npy_file):
    poses_bounds = np.load(npy_file)
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # 3x5 变换矩阵
    bounds = poses_bounds[:, -2:]  # 深度范围
    return poses, bounds

# 2. 读取 COLMAP sparse 文件夹中的相机数据
def load_colmap_data(sparse_folder):
    cameras_file = os.path.join(sparse_folder, 'cameras.bin')
    images_file = os.path.join(sparse_folder, 'images.bin')
    cameras = read_model.read_cameras_binary(cameras_file)
    images = read_model.read_images_binary(images_file)
    return cameras, images

# 3. 对相机位姿进行缩放
def scale_camera_positions(images, scale_factor):
    new_images = dict()
    for Idx in images.keys():
        original_image = images[Idx]
        new_images[Idx] = read_model.Image(original_image.id, original_image.qvec,
                                           original_image.tvec*scale_factor, 
                                           original_image.camera_id, original_image.name, original_image.xys, original_image.point3D_ids)

    return new_images

# 4. 保存修改后的数据到新的文件夹
def save_scaled_data(src_folder, dest_folder, cameras, images):
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
    # 复制无需修改的文件
    for file_name in ['cameras.bin', 'points3D.bin']:
        src_file = os.path.join(src_folder, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(dest_folder, file_name))

    # 保存修改后的 images.bin
    read_model.write_images_binary(images, os.path.join(dest_folder, 'images.bin'))

# 5. 可视化相机位置和朝向
def visualize_camera_poses(poses_bounds, original_images, scaled_images):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 可视化 poses_bounds.npy 中的相机
    for pose in poses_bounds:
        cam_pos = pose[:3, 3]
        cam_dir = pose[:3, 2]  # 相机的朝向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_dir[0], cam_dir[1], cam_dir[2], length=0.5, color='g')

    # 可视化缩放前的相机位置
    for image_id, image_data in original_images.items():
        cam_pos = -np.matmul(image_data.qvec2rotmat(), image_data.tvec)
        cam_dir = image_data.qvec2rotmat()[:, 2]  # 获取相机的 z 轴方向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_dir[0], cam_dir[1], cam_dir[2], length=0.5, color='r')

    # 可视化缩放后的相机位置
    for image_id, image_data in scaled_images.items():
        cam_pos = -np.matmul(image_data.qvec2rotmat(), image_data.tvec)
        cam_dir = image_data.qvec2rotmat()[:, 2]  # 获取相机的 z 轴方向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_dir[0], cam_dir[1], cam_dir[2], length=0.5, color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Poses Bounds', 'Original', 'Scaled'])
    plt.show()

# 6. 主函数执行流程
def main():

    
    # 读取 poses_bounds.npy 中的相机位姿和尺度信息
    poses_bounds, bounds = load_poses_bounds(npy_file)
    
    # 读取 sparse 文件夹中的数据
    cameras, original_images = load_colmap_data(sparse_folder)
    
    # 计算放缩因子
    scale_factor = np.mean(bounds[:, 1] - bounds[:, 0])  # 根据深度范围计算缩放因子
    
    # 对相机位置进行缩放
    scaled_images = scale_camera_positions(original_images.copy(), scale_factor)
    
    # 保存缩放后的相机数据到新的文件夹
    save_scaled_data(sparse_folder, new_sparse_folder, cameras, scaled_images)
    
    # 可视化 poses_bounds.npy、缩放前和缩放后的相机位置及方向
    visualize_camera_poses(poses_bounds, original_images, scaled_images)

if __name__ == '__main__':
    main()