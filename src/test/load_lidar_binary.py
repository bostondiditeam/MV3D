import numpy as np
import cv2

from kitti_data.pykitti import utils
from config import cfg
# from net.common import TOP_X_MIN,TOP_X_MAX,TOP_Y_MIN,TOP_Y_MAX
import data



TOP_Y_MIN=-20  #40
TOP_Y_MAX=+20
TOP_X_MIN=0
TOP_X_MAX=40   #70.4
TOP_Z_MIN=-2.0    ###<todo> determine the correct values!
TOP_Z_MAX= 0.4


TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1
TOP_Z_DIVISION=0.4


def lidar_to_top(lidar):

    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]


    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    quantized = np.dstack((qxs,qys,qzs,prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2
    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)


    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  #new method
        for z in range(Zn):
            iz = np.where (quantized[:,2]==z)
            quantized_z = quantized[iz]

            for y in range(Yn):
                iy  = np.where (quantized_z[:,1]==y)
                quantized_zy = quantized_z[iy]

                for x in range(Xn):
                    ix  = np.where (quantized_zy[:,0]==x)
                    quantized_zyx = quantized_zy[ix]
                    if len(quantized_zyx)>0:
                        yy,xx,zz = -x,-y, z

                        #height per slice
                        max_height = max(0,np.max(quantized_zyx[:,2])-TOP_Z_MIN)
                        top[yy,xx,zz]=max_height

                        #intensity
                        max_intensity = np.max(quantized_zyx[:,3])
                        top[yy,xx,Zn]=max_intensity

                        #density
                        count = len(idx)
                        top[yy,xx,Zn+1]+=count

                    pass
                pass
            pass

    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 0: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)


    return top, top_image

if 0:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    def plot_lidar(lidar):
        print(lidar.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(lidar)):
            ax.scatter(lidar[i,0], lidar[i,1], lidar[i,2],s=0.01,
                       c= [0.5,0.5,0.5])
            plt.axis('equal')
            # ax.set_xlabel('LabelX')
            # ax.set_ylabel('LabelY')
            # ax.set_zlabel('LabelZ')
            # ax.set_xlim(30, 40)
            # ax.set_ylim(-10, 10)
            # ax.set_zlim(0, 10)
        plt.show()


        lidars= utils.load_velo_scans(['30.bin'])
        intensity_max=np.max(lidars[0][:,3])

        print('shape: '+str(lidars[0].shape)+' intensity max: '+str(intensity_max))
        plot_lidar(lidars[0])
        top, top_image = data.lidar_to_top(lidars[0])
        cv2.imwrite('30.png', top_image)

if 0:
    top=np.load('1_t_15_00138.npy')

    top_image = np.sum(top, axis=2)
    top_image = top_image - np.min(top_image)
    top_image = (top_image / np.max(top_image) * 255)
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    cv2.imwrite('top_image_1.png', top_image)

    for i in range(8):
        top_image = top[:,:,i]
        top_image = top_image - np.min(top_image)
        top_image = (top_image / np.max(top_image) * 255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
        cv2.imwrite('top_image_{}.png'.format(i), top_image)

if 1:
    import mayavi.mlab as mlab

    from show_lidar import *

    lidars= utils.load_velo_scans(['kitti_005_0000000000.bin'])
    lidar=lidars[0]
    intensity_max=np.max(lidars[0][:,3])
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(500, 500))
    mlab.clf(fig)
    draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1)
    mlab.show()

    # print('shape: '+str(lidars[0].shape)+' intensity max: '+str(intensity_max))
    # plot_lidar(lidars[0])
    # top, top_image = data.lidar_to_top(lidars[0])
    # cv2.imwrite('30.png', top_image)


