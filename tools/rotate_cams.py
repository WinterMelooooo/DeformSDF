import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colmap_read_model as read_model  # 你之前提供的脚本处理 .bin 文件
from pathlib import Path
import json

# 1. 从 JSON 文件读取旋转矩阵和位置信息
def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    rotation_matrix = np.array(data['orientation'])  # 3x3 旋转矩阵
    position = np.array(data['position'])  # 3x1 平移向量
    return rotation_matrix, position

# 2. 读取 COLMAP sparse 文件夹中的相机数据
def load_colmap_data(sparse_folder):
    cameras_file = os.path.join(sparse_folder, 'cameras.bin')
    images_file = os.path.join(sparse_folder, 'images.bin')
    cameras = read_model.read_cameras_binary(cameras_file)
    images = read_model.read_images_binary(images_file)
    return cameras, images

# 3. 应用旋转矩阵并修改 images.bin 中的相机外参
def apply_rotation_to_images(images, rotation_matrix, position):
    new_images = dict()
    for Idx in images.keys():
        print(f"Img keys be:{Idx}")
        original_image = images[Idx]
        new_images[Idx] = read_model.Image(original_image.id, 
                                           read_model.rotmat2qvec(rotation_matrix), -np.matmul(rotation_matrix.T, position), 
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

# 5. 可视化相机位置和朝向（旋转前和旋转后）
def visualize_camera_poses(original_images, rotated_images, rotation_matrix, position):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    '''
    # 绘制旋转前的相机位置
    for image_id, image_data in original_images.items():
        cam_pos = -np.matmul(image_data.qvec2rotmat(), image_data.tvec)
        cam_dir = image_data.qvec2rotmat()[:, 2]  # 获取相机的 z 轴方向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_dir[0], cam_dir[1], cam_dir[2], length=0.5, color='r')
    '''
    # 绘制旋转后的相机位置
    for image_id, image_data in rotated_images.items():
        cam_pos = -np.matmul(image_data.qvec2rotmat(), image_data.tvec)
        cam_dir = image_data.qvec2rotmat()[:, 2]  # 获取相机的 z 轴方向
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_dir[0], cam_dir[1], cam_dir[2], length=0.5, color='g')

    # 可视化 JSON 中的旋转矩阵和位置信息
    ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 0], rotation_matrix[1, 0], rotation_matrix[2, 0], color='r', length=0.5)
    ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 1], rotation_matrix[1, 1], rotation_matrix[2, 1], color='b', length=0.5)
    ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 2], rotation_matrix[1, 2], rotation_matrix[2, 2], color='g', length=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Original', 'Rotated', 'JSON'])
    plt.show()

# 6. 主函数执行流程
def main():
    sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_Origin\sparse'  # 原始sparse文件夹路径
    new_sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Trash\sparse'  # 新的sparse文件夹路径
    json_file = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_XmZhu_Rotated\0_00296.json'  # JSON文件路径

    # 读取 JSON 中的相机旋转矩阵和位置
    rotation_matrix, position = load_json_data(json_file)

    # 读取 sparse 文件夹中的数据
    cameras, original_images = load_colmap_data(sparse_folder)

    # 应用旋转矩阵并修改相机外参
    rotated_images = apply_rotation_to_images(original_images, rotation_matrix, position)

    # 保存旋转后的相机数据到新的文件夹
    save_scaled_data(sparse_folder, new_sparse_folder, cameras, rotated_images)

    # 可视化相机位姿和朝向
    visualize_camera_poses(original_images, rotated_images, rotation_matrix, position)

if __name__ == '__main__':
    main()