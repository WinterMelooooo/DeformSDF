import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import colmap_read_model as read_model 
import collections
from scipy.spatial.transform import Rotation as R


sparse_folders = [r'D:\Melooooo\Lab\LLMM_2_DTU\Trash']
Camera_Pos_Rot = collections.namedtuple( "Camera_Pos_Rot", ['pos', 'rotation'] )
Colors = ['r','g','b']
ColorIdx = 0
 
 
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot



def Load_Rotation_and_Pos(path:str):
    images = read_model.read_images_binary(os.path.join(path,"images.bin"))
    Lst = list()
    for Idx in images.keys():
        image = images[Idx]
        Lst.append(Camera_Pos_Rot(-np.matmul(image.qvec2rotmat(), image.tvec), quaternion2rot(image.qvec)))
    print(f"Cam.rotaion.type() be: {type(Lst[0].rotation)}")
    print(f"There are {len(Lst)} Cams in CamLst")
    print(f"CamLst be:\n{Lst}")
    return Lst


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for folder in sparse_folders:
    CamLst = Load_Rotation_and_Pos(folder)
    for Cam in CamLst:
        ax.scatter(Cam.pos[0], Cam.pos[1], Cam.pos[2], c=Colors[ColorIdx], marker='o')
        '''
        ax.quiver(Cam.pos[0], Cam.pos[1], Cam.pos[2], #画x轴
                    Cam.pos[0]+Cam.rotation[0][0], 
                    Cam.pos[1]+Cam.rotation[1][0],
                    Cam.pos[2]+Cam.rotation[2][0],
                    color=Colors[ColorIdx])
        ax.quiver(Cam.pos[0], Cam.pos[1], Cam.pos[2], #画y轴
                    Cam.pos[0]+Cam.rotation[0][1], 
                    Cam.pos[1]+Cam.rotation[1][1],
                    Cam.pos[2]+Cam.rotation[2][1],
                    color=Colors[ColorIdx])
        ax.quiver(Cam.pos[0], Cam.pos[1], Cam.pos[2], #画z轴
                    Cam.pos[0]+Cam.rotation[0][2], 
                    Cam.pos[1]+Cam.rotation[1][2],
                    Cam.pos[2]+Cam.rotation[2][2],
                    color=Colors[ColorIdx])
        '''
    ColorIdx += 1
plt.show()
