import numpy as np
import os
import struct
from scipy.spatial.transform import Rotation as R
import colmap_read_model as read_model

def load_poses_bounds(npy_file):
    """加载 poses_bounds.npy 文件，提取相机位姿和深度边界"""
    data = np.load(npy_file)
    poses = data[:, :12].reshape(-1, 3, 4)  # 提取前 12 个数据（3x4 的矩阵，旋转和平移）
    bounds = data[:, 12:14]  # 提取最后两个数据（深度范围 near 和 far）
    return poses, bounds



def generate_images_bin(poses, original_images, output_folder):
    """生成 images.bin 文件"""
    images = {}
    for i, pose in enumerate(poses):
        print(i)
        # 从pose中提取旋转矩阵和平移向量
        R_mat = pose[:3, :3]  # 3x3旋转矩阵
        tvec = pose[:3, 3]  # 3x1平移向量
        
        # 旋转矩阵转为四元数
        qvec = read_model.rotmat2qvec(R_mat)
        original_image = original_images[i+1]
        # 创建image条目
        image_data = read_model.Image(original_image.id, 
                                    qvec, tvec, 
                                    original_image.camera_id, original_image.name, original_image.xys, original_image.point3D_ids)
        print(tvec,"\n")
        images[i + 1] = image_data

    # 写入到images.bin文件
    os.makedirs(output_folder, exist_ok=True)
    images_bin_path = os.path.join(output_folder, 'images.bin')
    read_model.write_images_binary(images, images_bin_path)
    print(f'images.bin has been saved to {images_bin_path}')

def load_colmap_data(sparse_folder):
    cameras_file = os.path.join(sparse_folder, 'cameras.bin')
    images_file = os.path.join(sparse_folder, 'images.bin')
    cameras = read_model.read_cameras_binary(cameras_file)
    images = read_model.read_images_binary(images_file)
    return cameras, images

# 主函数，执行流程
def main():
    npy_file = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_YpZhao_Resized\poses_bounds.npy'  # 替换为你的poses_bounds.npy文件路径
    sparse_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_Origin\sparse'
    output_folder = r'D:\Melooooo\Lab\LLMM_2_DTU\Trash'  # 替换为新的sparse文件夹路径

    # 从npy文件中加载相机位姿
    poses, bounds = load_poses_bounds(npy_file)
    cameras, original_images = load_colmap_data(sparse_folder)

    # 生成新的images.bin文件
    generate_images_bin(poses, original_images, output_folder)

if __name__ == '__main__':
    main()