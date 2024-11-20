import os
import json
import shutil

src_folder = r"/data/ymhe/spring_gaus/real_capture/dynamic" #should be prefix/dynamic
dest_folder = r"/data/yktang/spring_Gaus"
article_name = r"bun"

dest_folder = os.path.join(dest_folder, article_name)
seq_json = os.path.join(src_folder, "sequences", article_name, "0.json")
images_path = os.path.join(src_folder, "videos_images")

with open( seq_json, "r") as f:
    seq = json.load(f)
cam_names = [i for i in seq.keys() if i != "hit_frame"]
print(f"cam_names be: {cam_names}")
num_frames = len(seq[cam_names[0]])
for frame in range(num_frames):
    scan_folder = os.path.join(dest_folder, f"scan{frame}")
    dest = os.path.join(scan_folder, "image")
    os.makedirs(dest)
    for idx in range(len(cam_names)):
        cam_name = cam_names[idx]
        src_img = os.path.join(images_path, cam_name, seq[cam_name][frame])
        shutil.copy(src_img, dest)
        old_name = os.path.join(dest, seq[cam_name][frame])
        new_name = os.path.join(dest, "IMG{:03d}.jpg".format(idx))
        os.rename(old_name, new_name)
