import os
import colmap_read_model as read_model  # 使用你提供的 colmap_read_model 脚本

sparse_folder = r'D:\Melooooo\Lab\LLFF\data\sparse\0'

def load_colmap_data(sparse_folder):
    # 加载 cameras.bin, images.bin 文件
    cameras_file = os.path.join(sparse_folder, 'cameras.bin')
    images_file = os.path.join(sparse_folder, 'images.bin')
    
    # 读取相机参数和相机位姿
    cameras = read_model.read_cameras_binary(cameras_file)
    images = read_model.read_images_binary(images_file)
    
    return cameras, images

# 指定sparse文件夹的路径
cameras, images = load_colmap_data(sparse_folder)

# 打印出相机数量
print(f'COLMAP识别出 {len(images)} 台相机')