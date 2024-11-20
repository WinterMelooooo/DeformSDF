import os
import shutil
from datetime import datetime
import numpy as np
import open3d as o3d

# 使用示例
src_dir = r"/home/yktang/VolSDF/exps"  #Original folder of VolSDF/exps
surface_dir = r"/home/yktang/VolSDF/surfaces"  
'''
Should be like this:
    --surface_dir
    ----frame_0
    ------surface_2000.ply
    ----frame_1
    ------surface_2000.ply
    ----frame_2
    ------surface_2000.ply
'''
start_idx = 0  # 设置开始索引
end_idx = 9   # 设置结束索引
scale_factor = 0.7
translation_vec = [0.4, -0.84, 0]  # 平移向量，例如沿 x 方向平移 1.0，y 方向 2.0，z 方向 3.0
flip_x = True  # 是否沿 x 轴翻转
flip_y = False  # 是否沿 y 轴翻转
flip_z = False  # 是否沿 z 轴翻转


def copy_latest_ply_files(src_dir, dest_dir, start_idx, end_idx):
    for i in range(start_idx, end_idx + 1):
        dtu_folder = os.path.join(src_dir, f'dtu_{i}')
        
        if not os.path.exists(dtu_folder):
            print(f"文件夹 {dtu_folder} 不存在，跳过...")
            continue
        
        # 找到dtu_i下创建时间最晚的文件夹
        subfolders = [f for f in os.listdir(dtu_folder) if os.path.isdir(os.path.join(dtu_folder, f))]
        if not subfolders:
            print(f"文件夹 {dtu_folder} 下没有子文件夹，跳过...")
            continue
        
        # 根据文件夹名称（假设文件夹名称是时间格式），找到最晚的那个文件夹
        latest_folder = max(subfolders, key=lambda folder: datetime.strptime(folder, '%Y_%m_%d_%H_%M_%S'))
        latest_folder_path = os.path.join(dtu_folder, latest_folder)

        # 源文件路径
        ply_file_path = os.path.join(latest_folder_path, 'plots', 'surface_2000.ply')

        if not os.path.exists(ply_file_path):
            print(f"在 {latest_folder_path} 中找不到文件 surface_2000.ply，跳过...")
            continue

        # 创建目标目录
        frame_folder = os.path.join(dest_dir, f'frame_{i}')
        os.makedirs(frame_folder, exist_ok=True)

        # 复制文件
        dest_file_path = os.path.join(frame_folder, 'surface_2000.ply')
        shutil.copy(ply_file_path, dest_file_path)
        
        print(f"已复制 {ply_file_path} 到 {dest_file_path}")

def rescale_translate_and_flip_mesh(input_ply_path, output_ply_path, translation_vector, scale_factor, flip_x=False, flip_y=False, flip_z=False):
    # 读取带有面的 .ply 文件
    mesh = o3d.io.read_triangle_mesh(input_ply_path)

    # 检查是否读取成功
    if mesh.is_empty():
        print(f"无法读取文件: {input_ply_path}")
        return
    mesh.scale(scale_factor, center=mesh.get_center())

    # 创建平移矩阵
    translation = translation_vector

    # 对mesh中的顶点进行平移操作
    mesh.translate(translation)

    # 获取顶点的 numpy 数组
    vertices = np.asarray(mesh.vertices)

    # 获取法向量的 numpy 数组
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)

    # 检查是否需要沿 x 轴翻转
    if flip_x:
        vertices[:, 0] = -vertices[:, 0]
        if mesh.has_vertex_normals():
            normals[:, 0] = -normals[:, 0]  # 翻转法向量的 x 分量

    # 检查是否需要沿 y 轴翻转
    if flip_y:
        vertices[:, 1] = -vertices[:, 1]
        if mesh.has_vertex_normals():
            normals[:, 1] = -normals[:, 1]  # 翻转法向量的 y 分量

    # 检查是否需要沿 z 轴翻转
    if flip_z:
        vertices[:, 2] = -vertices[:, 2]
        if mesh.has_vertex_normals():
            normals[:, 2] = -normals[:, 2]  # 翻转法向量的 z 分量

    # 将修改后的顶点赋回到mesh中
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # 将修改后的法向量赋回到mesh中
    if mesh.has_vertex_normals():
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # 保存平移和翻转后的网格
    o3d.io.write_triangle_mesh(output_ply_path, mesh)
    print(f"已将平移和翻转后的网格保存到: {output_ply_path}")




#copy_latest_ply_files(src_dir, surface_dir, start_idx, end_idx)
surface_files = os.listdir(surface_dir)
for file in surface_files:
    file = os.path.join(surface_dir,file,"surface_2000.ply")
    rescale_translate_and_flip_mesh(file, file,translation_vec, scale_factor, flip_x, flip_y, flip_z)