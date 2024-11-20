import numpy as np
from plyfile import PlyData, PlyElement

# 加载 PLY 文件中的点和面
def load_ply(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vertices = np.array([list(vertex) for vertex in plydata['vertex']])
    faces = np.array([list(face[0]) for face in plydata['face']])
    return vertices, faces

# 加载 NPZ 文件中的点云
def load_npz(npz_file_path):
    data = np.load(npz_file_path)
    points = data['pos']  # 假设'pos'键存储的是(N, 3)的坐标数组
    return points

# 合并点云数据并保存到新的 PLY 文件
def save_combined_ply(output_file_path, vertices, faces):
    # 定义新 PLY 文件中的顶点
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices_array = np.array([tuple(v) for v in vertices], dtype=vertex_dtype)

    # 定义新 PLY 文件中的面
    face_dtype = [('vertex_indices', 'i4', (3,))]
    faces_array = np.array([(face,) for face in faces], dtype=face_dtype)

    # 创建 PlyElement
    vertex_element = PlyElement.describe(vertices_array, 'vertex')
    face_element = PlyElement.describe(faces_array, 'face')

    # 写入新的 PLY 文件
    PlyData([vertex_element, face_element], text=True).write(output_file_path)

# 主函数
def main():
    # 文件路径
    ply_file_path = '/home/yktang/VolSDF/exps/dtu_66/2024_10_08_22_24_48/plots/surface_2000.ply'  # 替换为你的 PLY 文件路径
    npz_file_path = '/home/yktang/Deformable3DGS/output/fluid_cut_RealWorldScale_YpZhao_002/vis/points3d_filtered_000.npz'  # 替换为你的 NPZ 文件路径
    output_file_path = '/home/yktang/Deformable3DGS/output/fluid_cut_RealWorldScale_YpZhao_002/combined_output.ply'  # 输出 PLY 文件的路径

    # 加载 PLY 文件和 NPZ 文件中的点数据
    ply_vertices, ply_faces = load_ply(ply_file_path)
    npz_points = load_npz(npz_file_path)
    print(npz_points.shape)
    print(ply_vertices.shape)
    # 合并点数据
    combined_vertices = np.vstack((ply_vertices, npz_points))
    print(combined_vertices.shape)
    # 保存为新的 PLY 文件
    save_combined_ply(output_file_path, combined_vertices, ply_faces)
    print(f"Combined PLY file saved to {output_file_path}")
    plydata = PlyData.read(output_file_path)


    vertices = plydata['vertex']
    faces = plydata['face']
    print(vertices)

if __name__ == "__main__":
    main()
