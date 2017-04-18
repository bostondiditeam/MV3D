import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from net.common import MATRIX_Mt,MATRIX_Kt
import net.processing.boxes3d as box3d


data_dir='/Users/zengxuefeng/Development/MV3D/src/test/'

def box3d_to_rgb_projections(boxes3d, Mt=None, Kt=None):

    if Mt is None: Mt = np.array(MATRIX_Mt)
    if Kt is None: Kt = np.array(MATRIX_Kt)

    num  = len(boxes3d)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for n in range(num):
        if 1:
            box3d = boxes3d[n]
            Ps = np.hstack(( box3d, np.ones((8,1))) )
            Qs = np.matmul(Ps,Mt)
            Qs = Qs[:,0:3]
            qs = np.matmul(Qs,Kt)
            zs = qs[:,2].reshape(8,1)
            qs = (qs/zs)
            projections[n] = qs[:, 0:2]
        else:
            _Kt = ([[200,      0., 0.],
                    [0.,      200, 0.],
                    [1368/2., 1096/2., 1.]])
            Kt = np.array(_Kt)
            box3d = boxes3d[n].copy()
            box3d[:,0],box3d[:,1],box3d[:,2]=box3d[:,1],box3d[:,2],box3d[:,0]
            Qs=box3d
            qs = np.matmul(Qs, Kt)
            zs = qs[:, 2].reshape(8, 1)
            qs = (qs / zs)
            projections[n] = qs[:, 0:2]
    print(projections)
    return projections

def draw_boxed3d_to_rgb(rgb, boxes3d):
    projections = box3d_to_rgb_projections(boxes3d)
    rgb = box3d.draw_rgb_projections(rgb, projections, color=(255, 0, 255), thickness=1)
    return rgb

def loadnpy(name):
    path=data_dir+name
    box3d=np.load(path)
#     print(box3d)
    print('{} shape = {}'.format(name,box3d.shape))
#     print(box3d[0,:,0])
    return box3d

def plotBox3d(box):
    print(box.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(box[:,0], box[:,1], box[:,2])
    plt.axis('equal')
    # ax.set_xlabel('LabelX')
    # ax.set_ylabel('LabelY')
    # ax.set_zlabel('LabelZ')
    # ax.set_xlim(30, 40)
    # ax.set_ylim(-10, 10)
    # ax.set_zlim(0, 10)
    plt.show()

def main():
    boxs = loadnpy('2011_09_26_0005_00110.npy')
    len(boxs)
    # for i in range(len(boxs)):
    #     plotBox3d(boxs[i])
    img=cv2.imread(data_dir+'2011_09_26_0005_00110.png')
    print(boxs[1:2,:,:])
    img_b=draw_boxed3d_to_rgb(img,boxs[:,:,:])
    # plotBox3d(boxs[:,:,:])
    cv2.imshow('img_b',img_b)
    cv2.waitKey()


if __name__ == '__main__':
    main()