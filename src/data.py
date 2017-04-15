from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *
import net.utility.draw as draw
from net.processing.boxes3d import *
import numpy
from config import cfg
import os
import cv2


# run functions --------------------------------------------------------------------------


## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label=0
        if obj.type=='Van' or obj.type=='Truck' or obj.type=='Car' or obj.type=='Tram':# todo : only  support 'Van'
            label = 1

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels


## lidar to top ##
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


## lidar to top ##
def lidar_to_top_old(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    ## start to make top  here !!!
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0


                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
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


def load(indexs, prefix):

    # fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    # data_dir=cfg.DATA_SETS_DIR
    data_seg = cfg.PREPROCESSED_DATA_SETS_DIR

    train_rgbs=[cv2.imread(os.path.join(data_seg,'rgb', prefix + '_%05d.png' % n),1) for n in indexs]
    train_tops=[np.load(os.path.join(data_seg,'top', prefix+'_%05d.npy' % n)) for n in indexs]
    train_fronts=[np.zeros((1, 1), dtype=np.float32) for n in indexs]
    train_gt_labels=[np.load(os.path.join(data_seg,'gt_labels', prefix+'_%05d.npy' % n)) for n in indexs]
    train_gt_boxes3d=[np.load(os.path.join(data_seg, 'gt_boxes3d', prefix+'_%05d.npy' % n)) for n in indexs]

    return train_rgbs,train_tops,train_fronts,train_gt_labels,train_gt_boxes3d

def getTopFeatureShape(top_shape,stride):
    return (top_shape[0]//stride, top_shape[1]//stride)

def getTopImages(indexs):
    data_dir = cfg.DATA_SETS_DIR
    return [ cv2.imread(os.path.join(data_dir, 'seg/top_image/top_image_%05d.png' % n), 1) for n in indexs]

def getLidarDatas(indexs):
    data_dir = cfg.DATA_SETS_DIR
    return [ np.load(os.path.join(data_dir, 'seg/lidar/lidar_%05d.npy' % n)) for n in indexs ]



# ## drawing ####
#
# def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):
#
#     pxs=lidar[:,0]
#     pys=lidar[:,1]
#     pzs=lidar[:,2]
#     prs=lidar[:,3]
#
#     if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
#
#     mlab.points3d(
#         pxs, pys, pzs, prs,
#         mode='point',  # 'point'  'sphere'
#         colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
#         scale_factor=1,
#         figure=fig)
#
#     #draw grid
#     if is_grid:
#         mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
#
#         for y in np.arange(-50,50,1):
#             x1,y1,z1 = -50, y, 0
#             x2,y2,z2 =  50, y, 0
#             mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
#
#         for x in np.arange(-50,50,1):
#             x1,y1,z1 = x,-50, 0
#             x2,y2,z2 = x, 50, 0
#             mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
#
#     #draw axis
#     if 1:
#         mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
#
#         axes=np.array([
#             [2.,0.,0.,0.],
#             [0.,2.,0.,0.],
#             [0.,0.,2.,0.],
#         ],dtype=np.float64)
#         fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
#             [20., 20., 0.,0.],
#             [20.,-20., 0.,0.],
#         ],dtype=np.float64)
#
#
#         mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
#         mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
#         mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
#         mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
#         mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
#
#     #draw top_image feature area
#     if is_top_region:
#         x1 = TOP_X_MIN
#         x2 = TOP_X_MAX
#         y1 = TOP_Y_MIN
#         y2 = TOP_Y_MAX
#         mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
#         mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
#         mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
#         mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
#
#
#
#     mlab.orientation_axes()
#     mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
#     print(mlab.view())
#
#
#
# def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):
#
#     num = len(gt_boxes3d)
#     for n in range(num):
#         b = gt_boxes3d[n]
#
#         mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
#         for k in range(0,4):
#
#             #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
#             i,j=k,(k+1)%4
#             mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
#
#             i,j=k+4,(k+1)%4 + 4
#             mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
#
#             i,j=k,k+4
#             mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
#
#     mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    raw_dir = cfg.RAW_DATA_SETS_DIR
    date  = '2011_09_26'
    drive = '0005'
    frames_index=[0,1,2, 3, 4, 5]
    frames_index = [0,]

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(raw_dir, date, drive,frames_index) #, range(0, 50, 5))

    # Load some data
    dataset.load_calib()         # Calibration data are accessible as named tuples
    # dataset.load_timestamps()    # Timestamps are parsed into datetime objects
    # dataset.load_oxts()          # OXTS packets are loaded as named tuples
    # dataset.load_gray()         # Left/right images are accessible as named tuples
    dataset.load_rgb()          # Left/right images are accessible as named tuples
    # dataset.load_left_rgb()
    dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]

    tracklet_file = os.path.join(dataset.data_path, 'tracklet_labels.xml')

    objects = read_objects(tracklet_file, frames_index)

    ############# convert   ###########################
    save_preprocess_dir = cfg.PREPROCESSED_DATA_SETS_DIR
    os.makedirs(save_preprocess_dir, exist_ok=True)

    if 1:  ## rgb images --------------------
        os.makedirs(save_preprocess_dir + '/rgb',exist_ok=True)
        count=0
        for n in frames_index:
            print('rgb images={}'.format(n))
            rgb = dataset.rgb[count][0]
            rgb =(rgb*255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if(cfg.DATA_SETS_TYPE =='didi'):
                pass # rgb = rgb[500:-80, :, :]
            elif cfg.DATA_SETS_TYPE =='kitti':
                pass
            else:
                raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))
            # rename it to something like 2011_09_26_0005_00000.png for kitti dataset.
            # In didi, it will be like 2_1_1_1490991690046339536.bin.png (means didi dataset 2, car 1, bag 1,
            # then timestamp)
            # todo fit it to didi dataset later.
            cv2.imwrite(save_preprocess_dir + '/rgb/'+date+'_'+drive+'_%05d.png'%n, rgb)
            # cv2.imwrite(save_preprocess_dir + '/rgb/rgb_%05d.png'%n,rgb)
            count+=1
        print('rgb image save done\n')


    if 1:  ## top view --------------------
        os.makedirs(save_preprocess_dir + '/lidar',exist_ok=True)
        os.makedirs(save_preprocess_dir + '/top',exist_ok=True)
        os.makedirs(save_preprocess_dir + '/top_image',exist_ok=True)

        for n in frames_index:
            print('top view={}'.format(n))
            lidar = dataset.velo[count]
            top, top_image = lidar_to_top(lidar)
            # rename it to something like 2011_09_26_0005_00000.npy for kitti dataset.
            # In didi, it will be like 2_1_1_1490991690046339536.npy (means didi dataset 2, car 1, bag 1,
            # then timestamp)
            np.save(save_preprocess_dir + '/lidar/'+date+'_'+drive+'_%05d.npy'%n,lidar)
            np.save(save_preprocess_dir + '/top/'+date+'_'+drive+'_%05d.npy'%n,top)
            cv2.imwrite(save_preprocess_dir + '/top_image/'+date+'_'+drive+'_%05d.png' % n, top_image)
            # cv2.imwrite(save_preprocess_dir + '/top_image/' top_image_%05d.png'%n,top_image)
        print('top view save done\n')




    if 1:  ## boxes3d  --------------------
        os.makedirs(save_preprocess_dir + '/gt_boxes3d',exist_ok=True)
        os.makedirs(save_preprocess_dir + '/gt_labels',exist_ok=True)
        count = 0
        for n in frames_index:
            print('boxes3d={}'.format(n))
            objs = objects[count]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)

            np.save(save_preprocess_dir + '/gt_boxes3d/'+date+'_'+drive+'_%05d.npy'%n,gt_boxes3d)
            np.save(save_preprocess_dir + '/gt_labels/'+date+'_'+drive+'_%05d.npy'%n,gt_labels)
            count += 1
    if 1: #dump gt boxes
        os.makedirs(save_preprocess_dir + '/gt_box_plot', exist_ok=True)
        count = 0
        for n in frames_index:
            print('rgb images={}'.format(n))
            rgb = dataset.rgb[count][0]
            rgb = (rgb * 255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if (cfg.DATA_SETS_TYPE == 'didi'):
                pass  # rgb = rgb[500:-80, :, :]
            elif cfg.DATA_SETS_TYPE == 'kitti':
                pass
            else:
                raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))
            objs = objects[count]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)
            img = draw.draw_boxed3d_to_rgb(rgb, gt_boxes3d)
            cv2.imwrite(save_preprocess_dir + '/gt_box_plot/'+date+'_'+drive+'_%05d.png'%n, img)
            count += 1
        print('gt box image save done\n')

    ############# analysis ###########################
    if 0: ## make mean
        mean_image = np.zeros((400,400),dtype=np.float32)
        frames_index=20
        for n in frames_index:
            print(n)
            top_image = cv2.imread(save_preprocess_dir + '/top_image/'+date+'_'+drive+'_%05d.npy'%n,0)
            mean_image += top_image.astype(np.float32)

        mean_image = mean_image / len(frames_index)
        cv2.imwrite(save_preprocess_dir + '/top_image/top_mean_image'+date+'_'+drive+'.png',mean_image)


    if 0: ## gt_3dboxes distribution ... location and box, height
        depths =[]
        aspects=[]
        scales =[]
        mean_image = cv2.imread(save_preprocess_dir + '/top_image/top_mean_image'+date+'_'+drive+'.png',0)

        for n in frames_index:
            print(n)
            gt_boxes3d = np.load(save_preprocess_dir + '/gt_boxes3d/'+date+'_'+drive+'_%05d.npy'%n)

            top_boxes = box3d_to_top_box(gt_boxes3d)
            draw_box3d_on_top(mean_image, gt_boxes3d,color=(255,255,255), thickness=1, darken=1)


            for i in range(len(top_boxes)):
                x1,y1,x2,y2 = top_boxes[i]
                w = math.fabs(x2-x1)
                h = math.fabs(y2-y1)
                area = w*h
                s = area**0.5
                scales.append(s)

                a = w/h
                aspects.append(a)

                box3d = gt_boxes3d[i]
                d = np.sum(box3d[0:4,2])/4 -  np.sum(box3d[4:8,2])/4
                depths.append(d)

        depths  = np.array(depths)
        aspects = np.array(aspects)
        scales  = np.array(scales)

        numpy.savetxt(save_preprocess_dir + '/depths'+date+'_'+drive+'.txt',depths)
        numpy.savetxt(save_preprocess_dir + '/aspects'+date+'_'+drive+'.txt',aspects)
        numpy.savetxt(save_preprocess_dir + '/scales'+date+'_'+drive+'.txt',scales)
        cv2.imwrite(save_preprocess_dir + '/top_image/top_rois'+date+'_'+drive+'.png', mean_image)


