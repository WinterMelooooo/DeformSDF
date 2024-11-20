from plyfile import PlyData

file_path = r"/home/yktang/VolSDF/exps/dtu_66/2024_10_08_22_24_48/plots/surface_2000.ply"

plydata = PlyData.read(file_path)


vertices = plydata['vertex']
faces = plydata['face']
count = 0
# 遍历每一个面
for i, face in enumerate(faces):
    vertex_indices = face['vertex_indices']
    vertex_coords = [(vertices[idx]['x'], vertices[idx]['y'], vertices[idx]['z']) for idx in vertex_indices]
    print(f"Face {i}:")
    for idx, coord in zip(vertex_indices, vertex_coords):
        print(f"  Vertex {idx}: {coord}")
    count += 1
    if count > 10:
        break