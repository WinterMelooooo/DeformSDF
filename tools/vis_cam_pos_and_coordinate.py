import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colmap_read_model as read_model  # 使用你之前提供的 colmap_read_model 处理 .bin 文件

# 1. 读取 poses_bounds.npy 文件中的相机位姿和尺度
def load_poses_bounds(npy_file):
    poses_bounds = np.load(npy_file)
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # 3x5 的矩阵
    return poses

# 2. 读取 COLMAP sparse 文件夹中的相机数据
def load_colmap_data(sparse_folder):
    images_file = os.path.join(sparse_folder, 'images.bin')
    images = read_model.read_images_binary(images_file)
    return images

# 3. 可视化相机位置和朝向
def visualize_camera_poses(poses_bounds, original_images, scaled_images):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 可视化 poses_bounds.npy 中的相机位置和朝向（绿色）
    for pose in poses_bounds:
        cam_pos = pose[:3, 3]  # 位置
        x_axis = pose[:3, 0]  # x 轴方向
        y_axis = pose[:3, 1]  # y 轴方向
        z_axis = pose[:3, 2]  # z 轴方向
        
        # 绘制相机位置
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='g', marker='o')
        
        # 绘制相机朝向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], x_axis[0], x_axis[1], x_axis[2], color='g', length=0.5)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=0.5)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], z_axis[0], z_axis[1], z_axis[2], color='g', length=0.5)

    # 可视化放缩前的 sparse/ 中的相机位置和朝向（红色）
    for image_id, image_data in original_images.items():
        cam_pos = -np.matmul(image_data.qvec2rotmat(), image_data.tvec)  # 计算相机位置
        x_axis = image_data.qvec2rotmat()[:, 0]  # x 轴方向
        y_axis = image_data.qvec2rotmat()[:, 1]  # y 轴方向
        z_axis = image_data.qvec2rotmat()[:, 2]  # z 轴方向
        
        # 绘制相机位置
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='r', marker='o')
        
        # 绘制相机朝向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=0.5)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], y_axis[0], y_axis[1], y_axis[2], color='r', length=0.5)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], z_axis[0], z_axis[1], z_axis[2], color='r', length=0.5)

    # 可视化放缩后的 sparse/ 中的相机位置和朝向（蓝色）
    for image_id, image_data in scaled_images.items():
        cam_pos = -np.matmul(image_data.qvec2rotmat(), image_data.tvec)  # 计算相机位置
        x_axis = image_data.qvec2rotmat()[:, 0]  # x 轴方向
        y_axis = image_data.qvec2rotmat()[:, 1]  # y 轴方向
        z_axis = image_data.qvec2rotmat()[:, 2]  # z 轴方向
        
        # 绘制相机位置
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='b', marker='o')
        
        # 绘制相机朝向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], x_axis[0], x_axis[1], x_axis[2], color='b', length=0.5)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], y_axis[0], y_axis[1], y_axis[2], color='b', length=0.5)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Poses Bounds', 'Original Sparse', 'Scaled Sparse'])
    plt.show()

# 4. 主函数执行流程
def main():
    npy_file = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_YpZhao_Resized\poses_bounds.npy'  # poses_bounds.npy 文件路径
    original_sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_Rotate\sparse'  # 原始sparse文件夹路径
    scaled_sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_Resized_after_Rotate\sparse'  # 放缩后的sparse文件夹路径

    # 读取 poses_bounds.npy 中的相机位姿
    poses_bounds = load_poses_bounds(npy_file)
    
    # 读取原始 sparse 文件夹中的数据
    original_images = load_colmap_data(original_sparse_folder)

    # 读取放缩后的 sparse 文件夹中的数据
    scaled_images = load_colmap_data(scaled_sparse_folder)
    
    # 可视化相机位置和朝向
    visualize_camera_poses(poses_bounds, original_images, scaled_images)

if __name__ == '__main__':
    main()
