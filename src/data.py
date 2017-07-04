from kitti_data import pykitti
# from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
# from kitti_data.draw import *
from kitti_data.io import *
import net.utility.draw as draw
from net.processing.boxes3d import *
from config import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
    TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION
from config import cfg
import os
import cv2
import numpy
import glob
from multiprocessing import Pool
from collections import OrderedDict
import config
import ctypes
from numba import jit
from matplotlib import pyplot as plt

import sys

if config.cfg.USE_CLIDAR_TO_TOP:
    so_path = os.path.join(os.path.split(__file__)[0],
                           "lidar_data_preprocess/Python_to_C_Interface/ver3/LidarTopPreprocess.so")
    print('here: ', so_path)
    assert (os.path.exists(so_path))
    SharedLib = ctypes.cdll.LoadLibrary(so_path)

class Preprocess(object):

    def __init__(self):

        self.labels_map = {
            #background
            'background':0,
            #car
            'Van':1,
            'Truck': 1,
            'Car': 1,
            'Tram': 1,
            #Pedestrian
            'Pedestrian':2
        }

        if config.cfg.SINGLE_CLASS_DETECTION==False:
            self.n_class = max([self.labels_map[key] for key in self.labels_map]) + 1
        else:
            self.n_class = 2

    @property
    def num_class(self):
        return self.n_class

    def rgb(self, rgb):
        rgb = crop_image(rgb)
        return rgb


    def bbox3d(self, obj):
        return box3d_compose(translation= obj.translation, rotation= obj.rotation, size= obj.size)


    def label(self, obj):
        if obj.type in self.labels_map.keys():
            label = self.labels_map[obj.type] if config.cfg.SINGLE_CLASS_DETECTION==False else 1
        else:
            label = self.labels_map['background']
        return label


    def lidar_to_top(self, lidar):
        if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test' or cfg.DATA_SETS_TYPE == 'didi2'):
            lidar = filter_center_car(lidar)

        if cfg.USE_CLIDAR_TO_TOP:
            top = clidar_to_top(lidar)
        else:
            top = lidar_to_top(lidar)

        return top


    # project 3D points to camera plane
    def project_points(self, points):
        pp = box3d_to_rgb_box(points)
        print(pp.shape)
        return pp


    # perspective transform camera image using bbox points
    def transform_image(self, img, bbox_src, bbox_dst):
        projected_pts_src = box3d_to_rgb_box(bbox_src)
        # print('projected_pts_src dim: ', projected_pts_src.shape)
        projected_pts_src = projected_pts_src.squeeze()
        projected_pts_dst = box3d_to_rgb_box(bbox_dst)
        # print('projected_pts_dst dim: ', projected_pts_dst.shape)
        projected_pts_dst = projected_pts_dst.squeeze()
        M1 = cv2.getPerspectiveTransform(np.float32(projected_pts_src[2:6]),
                                         np.float32(projected_pts_dst[2:6]))
        M2 = cv2.getPerspectiveTransform(np.float32(projected_pts_src[:4]),
                                         np.float32(projected_pts_dst[:4]))
        M = (M1 + M2) / 2
        rows, cols = img.shape[:2]
        new_img = cv2.warpPerspective(img, M, (cols, rows))
        return new_img


    def flip(self, rgb, top, boxes3d, axis=1):
        assert (axis == 0) or (axis == 1), "axis can be 0 or 1."

        # rgb, top, _, labels, boxes3d, f_id = val.load()
        rgb = rgb.squeeze()
        tops = top.squeeze()
        bbox = boxes3d

        bbox_new = np.copy(bbox)
        # self.display(tops, rgb, bbox)
        # bbox_new = bbox_new.squeeze()
        # (1, 8, 3)
        bbox_new[:, :, axis] = -bbox_new[:, :, axis]
        # print('bbox_new shape here: ', bbox_new.shape)
        # if axis == 1:
        #     rgb_new = self.transform_image(rgb, bbox, bbox_new)
        #     # rgb_new = cv2.flip(rgb, axis)
        # else:
        rgb_new = np.zeros(rgb.shape, dtype=np.int32)
        tops_new = cv2.flip(tops, axis)
        # self.display(tops_new, rgb_new, bbox_new)
        return (tops_new, rgb_new, bbox_new)

    # BGR to RGB conversion for opencv to matplotlib format
    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # project 3D points to camera plane
    def project_points(self, points):
        pp = box3d_to_rgb_box(points)
        print(pp.shape)
        return pp

    # draw bbox on camera image
    def drawBbox(self, img, corners, color=(255, 255, 0)):
        image = np.copy(img)
        thickness = 10
        for i in range(4):
            pt1, pt2 = corners[2 * i], corners[2 * i + 1]
            cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                     color=color, thickness=thickness)

            pt1, pt2 = corners[i + 2 * (i // 2)], corners[3 - i + 6 * (i // 2)]
            cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                     color=color, thickness=thickness)

            pt1, pt2 = corners[i], corners[i + 4]
            cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                     color=color, thickness=thickness)
        return image

    # display data
    def display(self, top_slices=None, camera_img=None, gt_bbox=None):
        if gt_bbox is not None:
            # print('gt_bbox size here: ', gt_bbox.shape)
            corners = gt_bbox  # .squeeze()
            # print('after squeeze: ', corners.shape)
            projected_corners = self.project_points(corners)
            projected_corners = projected_corners.squeeze()

            if camera_img is not None:
                camera_img = self.drawBbox(camera_img, projected_corners)

        fig_count = 0
        if camera_img is not None:
            fig_count += 1
            plt.figure(fig_count)
            plt.imshow(self.BGR2RGB(camera_img))
            plt.title('Camera image')
            plt.axis('off')
        else:
            print('No camera image available.')

        if top_slices is not None:
            fig_count += 1
            plt.figure(fig_count)
            n_height_maps = top_slices.shape[2]
            fig, axes = plt.subplots(1, n_height_maps, figsize=(30, 16))
            labels = ['height_' + str(i + 1) for i in range(n_height_maps - 2)] + ['Intensity', 'Density']

            for i, ax in enumerate(axes):
                top = top_slices[:, :, i]
                ax.imshow(top, cmap="hot")
                ax.set_title(labels[i])
                ax.axis('off')

preprocess = Preprocess()


@jit
def filter_center_car(lidar):
    idx = np.where(np.logical_or(numpy.abs(lidar[:, 0]) > 4.7/2, numpy.abs(lidar[:, 1]) > 2.1/2))
    lidar = lidar[idx]
    return lidar

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

def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top,axis=2)
    top_image = top_image-np.min(top_image)
    divisor = np.max(top_image)-np.min(top_image)
    top_image = (top_image/divisor*255)
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image


def clidar_to_top(lidar):
    # Calculate map size and pack parameters for top view and front view map (DON'T CHANGE THIS !)
    Xn = int((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION)
    Yn = int((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION)
    Zn = int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)

    top_flip = np.ones((Xn, Yn, Zn + 2), dtype=np.float32)  # DON'T CHANGE THIS !

    num = lidar.shape[0]  # DON'T CHANGE THIS !

    # call the C function to create top view maps
    # The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps
    SharedLib.createTopMaps(ctypes.c_void_p(lidar.ctypes.data),
                            ctypes.c_int(num),
                            ctypes.c_void_p(top_flip.ctypes.data),
                            ctypes.c_float(TOP_X_MIN), ctypes.c_float(TOP_X_MAX),
                            ctypes.c_float(TOP_Y_MIN), ctypes.c_float(TOP_Y_MAX),
                            ctypes.c_float(TOP_Z_MIN), ctypes.c_float(TOP_Z_MAX),
                            ctypes.c_float(TOP_X_DIVISION), ctypes.c_float(TOP_Y_DIVISION),
                            ctypes.c_float(TOP_Z_DIVISION),
                            ctypes.c_int(Xn), ctypes.c_int(Yn), ctypes.c_int(Zn)
                            )
    top = np.flipud(np.fliplr(top_flip))
    return top


@jit
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
    #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs=(pzs-TOP_Z_MIN)/TOP_Z_DIVISION
    quantized = np.dstack((qxs,qys,qzs,prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 2
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)


    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  #new method
        for x in range(Xn):
            ix  = np.where(quantized[:,0]==x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0 : continue
            yy = -x

            for y in range(Yn):
                iy  = np.where(quantized_x[:,1]==y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if  count==0 : continue
                xx = -y

                top[yy,xx,Zn+1] = min(1, np.log(count+1)/math.log(32))
                max_height_point = np.argmax(quantized_xy[:,2])
                top[yy,xx,Zn]=quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0 : continue
                    zz = z

                    #height per slice
                    max_height = max(0,np.max(quantized_xyz[:,2])-z)
                    top[yy,xx,zz]=max_height




    # if 0: #unprocess
    #     top_image = np.zeros((height,width,3),dtype=np.float32)
    #
    #     num = len(lidar)
    #     for n in range(num):
    #         x,y = qxs[n],qys[n]
    #         if x>=0 and x <width and y>0 and y<height:
    #             top_image[y,x,:] += 1
    #
    #     max_value=np.max(np.log(top_image+0.001))
    #     top_image = top_image/max_value *255
    #     top_image=top_image.astype(dtype=np.uint8)


    return top


def get_all_file_names(data_seg, dates, drivers):
    # todo: check if all files from lidar, rgb, gt_boxes3d is the same
    lidar_dir = os.path.join(data_seg, "top")
    load_indexs = []
    for date in dates:
        for driver in drivers:
            # file_prefix is something like /home/stu/data/preprocessed/didi/lidar/2011_09_26_0001_*
            file_prefix = lidar_dir + '/' + date + '_' + driver + '_*'
            driver_files = glob.glob(file_prefix)
            name_list = [file.split('/')[-1].split('.')[0] for file in driver_files]
            load_indexs += name_list
    return load_indexs


def proprecess_rgb(save_preprocess_dir,dataset,date,drive,frames_index,overwrite=False):

    dataset_dir=os.path.join(save_preprocess_dir,'rgb',date,drive)
    os.makedirs(dataset_dir, exist_ok=True)
    count = 0
    for n in frames_index:
        path=os.path.join(dataset_dir,'%05d.png' % n)
        if overwrite==False and os.path.isfile(path):
            count += 1
            continue
        print('rgb images={}'.format(n))
        rgb = dataset.rgb[count][0]
        rgb = preprocess.rgb(rgb)

        # todo fit it to didi dataset later.
        cv2.imwrite(os.path.join(path), rgb)
        # cv2.imwrite(save_preprocess_dir + '/rgb/rgb_%05d.png'%n,rgb)
        count += 1
    print('rgb image save done\n')

def generate_top_view(save_preprocess_dir,dataset,objects,date,drive,frames_index,
                      overwrite=False,dump_image=True):

    dataset_dir = os.path.join(save_preprocess_dir, 'top', date, drive)
    os.makedirs(dataset_dir, exist_ok=True)

    count = 0
    lidars=[]
    pool=Pool(3)
    for n in frames_index:
        path=os.path.join(dataset_dir,'%05d.npy' % n)
        if overwrite==False and os.path.isfile(path):
            count += 1
            continue
        lidars.append(dataset.velo[count])
        count += 1


    if cfg.USE_CLIDAR_TO_TOP:
        print('use clidar_to_top')
        t0 = time.time()
        tops = pool.map(clidar_to_top,lidars)
        # tops=[clidar_to_top(lidar) for lidar in lidars]
        print('time = ',time.time() -t0)
    else:
        t0 = time.time()
        tops = pool.map(lidar_to_top,lidars)
        # tops=[lidar_to_top(lidar) for lidar in lidars]
        print('time = ', time.time() - t0)

    count = 0
    for top in tops:
        n=frames_index[count]
        path = os.path.join(dataset_dir, '%05d.npy' % n)
        # top = top.astype(np.float16)
        # np.save(path, top)
        np.savez_compressed(path, top_view=top)
        print('top view {} saved'.format(n))
        count+=1


    if dump_image:
        dataset_dir = os.path.join(save_preprocess_dir, 'top_image', date, drive)
        os.makedirs(dataset_dir, exist_ok=True)

        top_images=pool.map(draw_top_image,tops)
        # top_images=[draw_top_image(top) for top in tops]

        count = 0
        for top_image in top_images:
            n = frames_index[count]
            top_image_path = os.path.join(dataset_dir,'%05d.png' % n)

            # draw bbox on top image
            if objects!=None:
                gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objects[count])
                top_image = draw_box3d_on_top(top_image, gt_boxes3d, color=(0, 0, 80))
            cv2.imwrite(top_image_path, top_image)
            count += 1
            print('top view image {} saved'.format(n))
    pool.close()
    pool.join()


def preprocess_bbox(save_preprocess_dir,objects,date,drive,frames_index,overwrite=False):

    bbox_dir = os.path.join(save_preprocess_dir, 'gt_boxes3d', date, drive)
    os.makedirs(bbox_dir, exist_ok=True)

    lable_dir = os.path.join(save_preprocess_dir, 'gt_labels', date, drive)
    os.makedirs(lable_dir, exist_ok=True)
    count = 0
    for n in frames_index:
        bbox_path=os.path.join(bbox_dir,'%05d.npy' % n)
        lable_path=os.path.join(lable_dir,'%05d.npy' % n)
        if overwrite==False and os.path.isfile(bbox_path):
            count += 1
            continue

        if overwrite==False and os.path.isfile(lable_path):
            count += 1
            continue

        print('boxes3d={}'.format(n))

        obj = objects[count]
        gt_boxes3d, gt_labels = obj_to_gt_boxes3d(obj)

        np.save(bbox_path, gt_boxes3d)
        np.save(lable_path, gt_labels)
        count += 1

def draw_top_view_image(save_preprocess_dir,objects,date,drive,frames_index,overwrite=False):

    dataset_dir = os.path.join(save_preprocess_dir, 'top_image', date, drive)
    os.makedirs(dataset_dir, exist_ok=True)
    count = 0
    for n in frames_index:
        top_image_path=os.path.join(dataset_dir,'%05d.png' % n)
        if overwrite==False and os.path.isfile(top_image_path):
            count += 1
            continue

        print('draw top view image ={}'.format(n))

        top = np.load(os.path.join(save_preprocess_dir,'top',date,drive,'%05d.npy.npz' % n) )
        top = top['top_view']
        top_image = draw_top_image(top)

        # draw bbox on top image
        if objects != None:
            gt_boxes3d = np.load(os.path.join(save_preprocess_dir,'gt_boxes3d',date,drive,'%05d.npy' % n))
            top_image = draw_box3d_on_top(top_image, gt_boxes3d, color=(0, 0, 80))
        else:
            print('Not found gt_boxes3d,skip draw bbox on top image')

        cv2.imwrite(top_image_path, top_image)
        count += 1
    print('top view image draw done\n')

def dump_lidar(save_preprocess_dir,dataset,date,drive,frames_index,overwrite=False):

    dataset_dir = os.path.join(save_preprocess_dir, 'lidar', date, drive)
    os.makedirs(dataset_dir, exist_ok=True)
    count = 0
    for n in frames_index:

        lidar_dump_path=os.path.join(dataset_dir,'%05d.npy' % n)
        if overwrite==False and os.path.isfile(lidar_dump_path):
            count += 1
            continue

        print('lidar data={}'.format(n))
        lidar = dataset.velo[count]
        np.save(lidar_dump_path, lidar)
        count += 1
    print('dump lidar data done\n')

def dump_bbox_on_camera_image(save_preprocess_dir,dataset,objects,date,drive,frames_index,overwrite=False):
    dataset_dir = os.path.join(save_preprocess_dir, 'gt_box_plot', date, drive)
    os.makedirs(dataset_dir, exist_ok=True)
    count = 0
    for n in frames_index:
        print('rgb images={}'.format(n))
        rgb = dataset.rgb[count][0]
        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb=crop_image(rgb)

        objs = objects[count]
        gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)
        img = draw.draw_box3d_on_camera(rgb, gt_boxes3d)
        new_size = (img.shape[1] // 3, img.shape[0] // 3)
        img = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(dataset_dir,'%05d.png' % n), img)
        count += 1
    print('gt box image save done\n')

def crop_image(image):
    image_crop=image.copy()
    left=cfg.IMAGE_CROP_LEFT  #pixel
    right=cfg.IMAGE_CROP_RIGHT
    top=cfg.IMAGE_CROP_TOP
    bottom=cfg.IMAGE_CROP_BOTTOM
    bottom_index= -bottom if bottom!= 0 else None
    right_index = -right if right != 0 else None
    image_crop=image_crop[top:bottom_index,left:right_index,:]
    return image_crop

def is_evaluation_dataset(date, drive):
    if date=='Round1Test' or date == 'test_car' or date == 'test_ped':
        return True
    else:
        return False

def data_in_single_driver(raw_dir, date, drive, frames_index=None):

    if (cfg.DATA_SETS_TYPE == 'didi2'):
        img_path = os.path.join(raw_dir, date, drive, "image_02", "data")
    elif (cfg.DATA_SETS_TYPE == 'didi'):
        img_path = os.path.join(raw_dir, date, drive, "image_02", "data")
    elif cfg.DATA_SETS_TYPE == 'kitti':
        img_path = os.path.join(raw_dir, date, date + "_drive_" + drive + "_sync", "image_02", "data")
    elif(cfg.DATA_SETS_TYPE == 'test'):
        img_path = os.path.join(raw_dir, date, drive, "image_02", "data")
    else:
        raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))

    if frames_index is None:
        nb_frames = len(glob.glob(img_path+"/*.png"))
        frames_index = range(nb_frames)

    # spilt large numbers of frame to small chunks
    if (cfg.DATA_SETS_TYPE == 'test'):
        max_cache_frames_num = 3
    else:
        max_cache_frames_num = 3
    if len(frames_index)>max_cache_frames_num:
        frames_idx_chunks=[frames_index[i:i+max_cache_frames_num] for i in range(0,len(frames_index),max_cache_frames_num)]
    else:
        frames_idx_chunks=[frames_index]

    for i, frames_index in enumerate(frames_idx_chunks):
        # The range argument is optional - default is None, which loads the whole dataset
        dataset = pykitti.raw(raw_dir, date, drive, frames_index) #, range(0, 50, 5))

        # read objects
        tracklet_file = os.path.join(dataset.data_path, 'tracklet_labels.xml')
        if os.path.isfile(tracklet_file):
            objects = read_objects(tracklet_file, frames_index)
        elif is_evaluation_dataset(date, drive):
            objects=None
            print("Skip read evaluation_dataset's tracklet_labels file : ")
        else:
            raise ValueError('read_objects error!!!!!')

        # Load some data
        # dataset.load_calib()         # Calibration data are accessible as named tuples
        # dataset.load_timestamps()    # Timestamps are parsed into datetime objects
        # dataset.load_oxts()          # OXTS packets are loaded as named tuples
        # dataset.load_gray()         # Left/right images are accessible as named tuples
        dataset.load_left_rgb()
        dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]


        ############# convert   ###########################
        save_preprocess_dir = cfg.PREPROCESSING_DATA_SETS_DIR

        if 1:  ## rgb images --------------------
            proprecess_rgb(save_preprocess_dir, dataset, date, drive, frames_index, overwrite=False)


        if 1:  ##generate top view --------------------
            generate_top_view(save_preprocess_dir, dataset,objects, date, drive, frames_index,
                              overwrite=True,dump_image=True)

        if 1 and objects!=None:  ## preprocess boxes3d  --------------------
            preprocess_bbox(save_preprocess_dir, objects, date, drive, frames_index, overwrite=True)

        if 0: ##draw top image with bbox
            draw_top_view_image(save_preprocess_dir, objects, date, drive, frames_index, overwrite=True)


        # dump lidar data
        if 0:
            dump_lidar(save_preprocess_dir, dataset, date, drive, frames_index, overwrite=False)

        if 1 and objects!= None: #dump gt boxes
            dump_bbox_on_camera_image(save_preprocess_dir, dataset, objects, date, drive, frames_index, overwrite=True)

        ############# analysis ###########################
        # if 0: ## make mean
        #     mean_image = np.zeros((400,400),dtype=np.float32)
        #     frames_index=20
        #     for n in frames_index:
        #         print(n)
        #         top_image = cv2.imread(save_preprocess_dir + '/top_image/'+date+'_'+drive+'_%05d.npy'%n,0)
        #         mean_image += top_image.astype(np.float32)
        #
        #     mean_image = mean_image / len(frames_index)
        #     cv2.imwrite(save_preprocess_dir + '/top_image/top_mean_image'+date+'_'+drive+'.png',mean_image)
        #
        #
        # if 0: ## gt_3dboxes distribution ... location and box, height
        #     depths =[]
        #     aspects=[]
        #     scales =[]
        #     mean_image = cv2.imread(save_preprocess_dir + '/top_image/top_mean_image'+date+'_'+drive+'.png',0)
        #
        #     for n in frames_index:
        #         print(n)
        #         gt_boxes3d = np.load(save_preprocess_dir + '/gt_boxes3d/'+date+'_'+drive+'_%05d.npy'%n)
        #
        #         top_boxes = box3d_to_top_box(gt_boxes3d)
        #         draw_box3d_on_top(mean_image, gt_boxes3d,color=(255,255,255), thickness=1, darken=1)
        #
        #         for i in range(len(top_boxes)):
        #             x1,y1,x2,y2 = top_boxes[i]
        #             w = math.fabs(x2-x1)
        #             h = math.fabs(y2-y1)
        #             area = w*h
        #             s = area**0.5
        #             scales.append(s)
        #
        #             a = w/h
        #             aspects.append(a)
        #
        #             box3d = gt_boxes3d[i]
        #             d = np.sum(box3d[0:4,2])/4 -  np.sum(box3d[4:8,2])/4
        #             depths.append(d)
        #
        #     depths  = np.array(depths)
        #     aspects = np.array(aspects)
        #     scales  = np.array(scales)
        #
        #     numpy.savetxt(save_preprocess_dir + '/depths'+date+'_'+drive+'.txt',depths)
        #     numpy.savetxt(save_preprocess_dir + '/aspects'+date+'_'+drive+'.txt',aspects)
        #     numpy.savetxt(save_preprocess_dir + '/scales'+date+'_'+drive+'.txt',scales)
        #     cv2.imwrite(save_preprocess_dir + '/top_image/top_rois'+date+'_'+drive+'.png', mean_image)


def preproces(mapping, frames_index):
    # if mapping is none, using all dataset under raw_data_sets_dir.
    if mapping is None:
        paths = glob.glob(os.path.join(cfg.RAW_DATA_SETS_DIR ,'*'))
        map_key = [os.path.basename(path) for path in paths]
        map_value = [os.listdir(bag_name) for bag_name in map_key]
        mapping = {k: v for k, v in zip(map_key, map_value)}

    # looping through
    for key in mapping.keys():
        if mapping[key] is None:
            paths = glob.glob(os.path.join(cfg.RAW_DATA_SETS_DIR, key, '*'))
            if len(paths) == 0:
                raise ValueError('can not found any file in:{}'.format(os.path.join(cfg.RAW_DATA_SETS_DIR, key, '*')))
            drivers_des=[os.path.basename(path) for path in paths]
        else:
            drivers_des=mapping[key]
        for driver in drivers_des:
            print('date {} and driver {}'.format(key, driver))
            data_in_single_driver(cfg.RAW_DATA_SETS_DIR, key, driver, frames_index)

# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    if (cfg.DATA_SETS_TYPE == 'didi'):
        data_dir = {'1': ['15', '10']}
        data_dir = OrderedDict(data_dir)
        frames_index = None  # None
    elif (cfg.DATA_SETS_TYPE == 'didi2'):
        dir_prefix = '/home/stu/round12_data/raw/didi'

        bag_groups = ['suburu_pulling_to_left',
                 'nissan_following_long',
                 'suburu_following_long',
                 'nissan_pulling_to_right',
                 'suburu_not_visible',
                 'cmax_following_long',
                 'nissan_driving_past_it',
                 'cmax_sitting_still',
                 'suburu_pulling_up_to_it',
                 'suburu_driving_towards_it',
                 'suburu_sitting_still',
                 'suburu_driving_away',
                 'suburu_follows_capture',
                 'bmw_sitting_still',
                 'suburu_leading_front_left',
                 'nissan_sitting_still',
                 'nissan_brief',
                 'suburu_leading_at_distance',
                 'bmw_following_long',
                 'suburu_driving_past_it',
                 'nissan_pulling_up_to_it',
                 'suburu_driving_parallel',
                 'nissan_pulling_to_left',
                 'nissan_pulling_away', 'ped_train']

        bag_groups = ['suburu_pulling_to_left',
                     'nissan_following_long',
                     'nissan_driving_past_it',
                     'cmax_sitting_still',
                      'cmax_following_long',
                     'suburu_driving_towards_it',
                     'suburu_sitting_still',
                     'suburu_driving_away',
                     'suburu_follows_capture',
                     'bmw_sitting_still',
                     'suburu_leading_front_left',
                     'nissan_sitting_still',
                     'suburu_leading_at_distance',
                     'suburu_driving_past_it',
                     'nissan_pulling_to_left',
                     'nissan_pulling_away', 'ped_train']

        # use orderedDict to fix the dictionary order.
        data_dir = OrderedDict([(bag_group, None) for bag_group in bag_groups])
        print('ordered dictionary here: ', data_dir)

        frames_index=None  #None
    elif cfg.DATA_SETS_TYPE == 'kitti':
        data_dir = {'2011_09_26': ['0001', '0017', '0029', '0052', '0070', '0002', '0018', '0035', '0056', '0079',
                                   '0019', '0036', '0005', '0057', '0084', '0020', '0039', '0059', '0086', '0011',
                                   '0023', '0046', '0060', '0091','0013', '0027', '0048', '0061', '0015', '0028',
                                   '0051', '0064']}

        frames_index = None # [0,5,8,12,16,20,50]
    elif cfg.DATA_SETS_TYPE == 'test':
        data_dir = {'1':None, '2':None}
        data_dir = OrderedDict(data_dir)
        frames_index=None
    else:
        raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))

    import time
    t0=time.time()

    preproces(data_dir, frames_index)

    print('use time : {}'.format(time.time()-t0))




