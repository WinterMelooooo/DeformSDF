import numpy as np
import struct
import os
import colmap_read_model as read_model
from scipy.spatial.transform import Rotation

images_file = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_XmZhu_Origin_Not_Rotated_2_DTU\sparse\0\images.bin'
cameras_file = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_XmZhu_Origin_Not_Rotated_2_DTU\sparse\0\cameras.bin'
dest = r'D:\Melooooo\Lab\LLMM_2_DTU\Cam_LLFF_XmZhu_Origin_Not_Rotated_2_DTU\sparse\0\bbs.npz'
camera_id = 1
width = 1920
height = 1200

def calculate_bounding_box(qvec, tvec, f, cx, cy):
    """计算相机边界框"""
    # 计算旋转矩阵
    R = read_model.qvec2rotmat(qvec)
    
    # 计算相机在世界坐标系中的位置
    cam_pos = tvec
    
    # 计算视野范围（假设使用标准视场角）
    # 这里可以使用更复杂的方法计算，具体取决于需要
    half_width = cx
    half_height = cy
    
    # 视锥体的角点
    points = np.array([
        [half_width, half_height, -f],
        [-half_width, half_height, -f],
        [half_width, -half_height, -f],
        [-half_width, -half_height, -f]
    ])
    
    # 旋转和移动
    bounding_box = (R @ points.T).T + cam_pos
    
    # 计算边界框的最小和最大点
    min_point = bounding_box.min(axis=0)
    max_point = bounding_box.max(axis=0)
    
    return min_point, max_point

def generate_bbs_npz(images, cameras, output_file):
    """生成 bbs.npz 文件"""
    bbs = []
    
    for image_id in images.keys():
        data = images[image_id]
        qvec = data.qvec
        tvec = data.tvec
        
        # 从相机参数中获取内参
        fx, fy, cx, cy = cameras[camera_id].params  # 需要从 cameras 中获取相机参数
        
        # 计算边界框
        min_point, max_point = calculate_bounding_box(qvec, tvec, fx, cx, cy)
        print([min_point,max_point])
        bbs.append(np.concatenate([min_point, max_point]))

    # 保存为 npz 文件
    np.savez_compressed(output_file, bbs=np.array(bbs))

# 示例主函数
def main():
    # 替换为你的文件路径
    
    # 读取相机参数
    images = read_model.read_images_binary(images_file)
    cameras = read_model.read_cameras_binary(cameras_file)  # 需要实现这个函数读取 cameras.bin
    
    # 生成 bbs.npz
    generate_bbs_npz(images, cameras, dest)

if __name__ == "__main__":
    main()
