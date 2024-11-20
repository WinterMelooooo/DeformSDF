import numpy as np
import json
import cv2

cameras_json_path = "/data/ymhe/spring_gaus/real_capture/dynamic/cameras_calib.json"
dest_npz_path = "/data/yktang/spring_Gaus/cameras.npz"

# 加载 cameras_calib.json 文件
with open(cameras_json_path, 'r') as f:
    calib_data = json.load(f)

# 提取内参矩阵
camera_intrinsics = np.array(calib_data["camera_matrix"])

cam_dict = {}

cam_names = ["C0733", "C0787", "C0801"]
# 提取和转换每个相机的旋转和平移向量
for id in range(len(cam_names)):
    cam_id = cam_names[id]
    rvec = np.array(calib_data[cam_id]["rvecs"]).reshape(3)
    tvec = np.array(calib_data[cam_id]["tvecs"]).reshape(3, 1)

    # 使用 Rodrigues 公式将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # camera_intrinsics 是 3x3 内参矩阵， R 是 3x3 旋转矩阵， tvec 是 3x1 平移向量
    P = camera_intrinsics @ np.hstack((R, tvec))

    # 将 P 扩展为 4x4 的 world_matrix
    world_matrix = np.vstack((P, [0, 0, 0, 1]))

    cam_dict[f"world_mat_{id}"] = world_matrix

np.savez(dest_npz_path, **cam_dict)

file = np.load(dest_npz_path)
print(file.files)
for name in file.files:
    print(file[name])