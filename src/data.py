from kitti_data import pykitti
# from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
# from kitti_data.draw import *
from kitti_data.io import *
import net.utility.draw as draw
from net.processing.boxes3d import *
from net.common import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
    TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION
from config import cfg
import os
import cv2
import numpy
import glob
from multiprocessing import Pool


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
    top_image = (top_image/np.max(top_image)*255)
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image


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

    if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test'):
        lidar=filter_center_car(lidar)


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
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
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
        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb = crop_image(rgb)

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

    tops = pool.map(lidar_to_top,lidars)
    # tops=[lidar_to_top(lidar) for lidar in lidars]
    count = 0
    for top in tops:
        n=frames_index[count]
        path = os.path.join(dataset_dir, '%05d.npy' % n)
        np.save(path, top)
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

        top = np.load(os.path.join(save_preprocess_dir,'top',date,drive,'%05d.npy' % n) )
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
        if (cfg.DATA_SETS_TYPE == 'didi'):
            pass  # rgb = rgb[500:-80, :, :]
        elif cfg.DATA_SETS_TYPE == 'kitti':
            pass
        elif cfg.DATA_SETS_TYPE == 'test':
            pass
        else:
            raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))
        objs = objects[count]
        gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)
        img = draw.draw_boxed3d_to_rgb(rgb, gt_boxes3d)
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
    if date=='Round1Test':
        return True
    else:
        return False

def data_in_single_driver(raw_dir, date, drive, frames_index=None):

    if (cfg.DATA_SETS_TYPE == 'didi'):
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
        max_cache_frames_num = 4
    else:
        max_cache_frames_num = 4
    if len(frames_index)>max_cache_frames_num:
        frames_idx_chunks=[frames_index[i:i+max_cache_frames_num] for i in range(0,len(frames_index),max_cache_frames_num)]
    else:
        frames_idx_chunks=[frames_index]

    for frames_index in frames_idx_chunks:
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

        if 0 and objects!= None: #dump gt boxes
            dump_bbox_on_camera_image(save_preprocess_dir, dataset, objects, date, drive, frames_index, overwrite=False)

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


def preproces(dates=None, drivers=None, frames_index=None):

    if dates == None:
        paths = glob.glob(os.path.join(cfg.RAW_DATA_SETS_DIR ,'*'))
        dates = [os.path.basename(path) for path in paths]
    for date in dates:
        if drivers==None:
            paths = glob.glob(os.path.join(cfg.RAW_DATA_SETS_DIR,date,'*'))
            if paths==[]:
                raise ValueError('can not found any file in:{}'.format(os.path.join(cfg.RAW_DATA_SETS_DIR,date,'*')))
            drivers_searched = [os.path.basename(path) for path in paths]
            drivers_des=drivers_searched
        else:
            drivers_des=drivers
        for driver in drivers_des:
            data_in_single_driver(cfg.RAW_DATA_SETS_DIR, date, driver, frames_index)

# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    if (cfg.DATA_SETS_TYPE == 'didi'):
        #dates=['1','2','3']
        dates = ['Round1Test']
        drivers= None
        frames_index=None
    elif cfg.DATA_SETS_TYPE == 'kitti':
        dates = ['2011_09_26']
        drivers = ['0001', '0017', '0029', '0052', '0070', '0002', '0018', '0035', '0056', '0079', '0005', '0019',
                   '0036',
                   '0057', '0084', '0009', '0020', '0039', '0059', '0086', '0011', '0023', '0046', '0060', '0091',
                   '0013', '0027', '0048',
                   '0061', '0015', '0028', '0051', '0064']
        drivers=['0005']
        frames_index=[0,5,8,12,16,20,50]
    elif cfg.DATA_SETS_TYPE == 'test':
        dates = ['1','2']
        drivers = None
        frames_index=None
    else:
        raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))

    import time
    t0=time.time()

    preproces(dates, drivers, frames_index)

    print('use time : {}'.format(time.time()-t0))




