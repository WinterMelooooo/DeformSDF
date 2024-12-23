import os
from glob import glob
import torch
import re
import trimesh
import open3d as o3d
import numpy as np

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def split_pnt_cloud(pnt_cloud, n_points=1000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(pnt_cloud.shape[0]).cuda(), n_points, dim=0)):
        split.append(pnt_cloud[indx])
    return split

def merge_pnt_cloud(res):
    ''' Merge the split output. '''
    return torch.cat(res, 0)


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_scans(folder_path):
    min_frame = float('inf')
    max_frame = float('-inf')
    
    # 正则表达式匹配 "scan" 后跟随数字的格式
    pattern = re.compile(r'scan(\d+)$')

    # 遍历父文件夹中的所有文件夹
    for folder_name in os.listdir(folder_path):
        match = pattern.match(folder_name)
        if match:
            frame_num = int(match.group(1))
            min_frame = min(min_frame, frame_num)
            max_frame = max(max_frame, frame_num)

    # 如果没有找到符合条件的文件夹
    if min_frame == float('inf') or max_frame == float('-inf'):
        return None, None
    
    return [i for i in range(min_frame, max_frame+1)]

def sample_point_cloud_from_surface_mesh( mesh:trimesh.Trimesh, num_points = 5000, init_factor = 5):
    vertices = np.array(mesh.vertices)  
    faces = np.array(mesh.faces)  
    mesh = o3d.geometry.TriangleMesh() 
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor = 5)
    mesh_area = mesh.get_surface_area()
    average_area = mesh_area / len(pcd.points)

    # 估算半径
    radius = (average_area / 3.14159) ** 0.5
    return pcd, radius