from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *

from net.utility.draw import *
from net.processing.boxes3d import *


# run functions --------------------------------------------------------------------------





## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label = 1 #<todo>

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels


## lidar to top ##
def lidar_to_top(lidar):

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


## drawing ####

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    print(mlab.view())



def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    basedir = '/root/share/project/didi/data/kitti/dummy'
    date  = '2011_09_26'
    drive = '0005'

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive) #, range(0, 50, 5))

    # Load some data
    dataset.load_calib()         # Calibration data are accessible as named tuples
    dataset.load_timestamps()    # Timestamps are parsed into datetime objects
    dataset.load_oxts()          # OXTS packets are loaded as named tuples
    #dataset.load_gray()         # Left/right images are accessible as named tuples
    #dataset.load_rgb()          # Left/right images are accessible as named tuples
    dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]

    tracklet_file = '/root/share/project/didi/data/kitti/dummy/2011_09_26/tracklet_labels.xml'

    num_frames=len(dataset.velo)  #154
    objects = read_objects(tracklet_file, num_frames)

    ############# convert   ###########################
    os.makedirs('/root/share/project/didi/data/kitti/dummy/seg',exist_ok=True)

    if 0:  ## rgb images --------------------
        os.makedirs('/root/share/project/didi/data/kitti/dummy/seg/rgb',exist_ok=True)

        for n in range(num_frames):
            print(n)
            rgb = dataset.rgb[n][0]
            rgb =(rgb*255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite('/root/share/project/didi/data/kitti/dummy/seg/rgb/rgb_%05d.png'%n,rgb)

        exit(0)


    if 0:  ## top images --------------------
        os.makedirs('/root/share/project/didi/data/kitti/dummy/seg/lidar',exist_ok=True)
        os.makedirs('/root/share/project/didi/data/kitti/dummy/seg/top',exist_ok=True)
        os.makedirs('/root/share/project/didi/data/kitti/dummy/seg/top_image',exist_ok=True)

        for n in range(num_frames):
            print(n)
            lidar = dataset.velo[n]
            top, top_image = lidar_to_top(lidar)

            np.save('/root/share/project/didi/data/kitti/dummy/seg/lidar/lidar_%05d.npy'%n,lidar)
            np.save('/root/share/project/didi/data/kitti/dummy/seg/top/top_%05d.npy'%n,top)
            cv2.imwrite('/root/share/project/didi/data/kitti/dummy/seg/top_image/top_image_%05d.png'%n,top_image)

        exit(0)



    if 1:  ## boxes3d  --------------------
        os.makedirs('/root/share/project/didi/data/kitti/dummy/seg/gt_boxes3d',exist_ok=True)
        os.makedirs('/root/share/project/didi/data/kitti/dummy/seg/gt_labels',exist_ok=True)
        for n in range(num_frames):
            print(n)
            objs = objects[n]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)

            np.save('/root/share/project/didi/data/kitti/dummy/seg/gt_boxes3d/gt_boxes3d_%05d.npy'%n,gt_boxes3d)
            np.save('/root/share/project/didi/data/kitti/dummy/seg/gt_labels/gt_labels_%05d.npy'%n,gt_labels)

        exit(0)


    ############# analysis ###########################
    if 0: ## make mean
        mean_image = np.zeros((400,400),dtype=np.float32)
        num_frames=20
        for n in range(num_frames):
            print(n)
            top_image = cv2.imread('/root/share/project/didi/data/kitti/dummy/seg/top_image/top_image_%05d.png'%n,0)
            mean_image += top_image.astype(np.float32)

        mean_image = mean_image/num_frames
        cv2.imwrite('/root/share/project/didi/data/kitti/dummy/seg/top_image/top_mean_image.png',mean_image)


    if 0: ## gt_3dboxes distribution ... location and box, height
        depths =[]
        aspects=[]
        scales =[]
        mean_image = cv2.imread('/root/share/project/didi/data/kitti/dummy/seg/top_image/top_mean_image.png',0)

        for n in range(num_frames):
            print(n)
            gt_boxes3d = np.load('/root/share/project/didi/data/kitti/dummy/seg/gt_boxes3d/gt_boxes3d_%05d.npy'%n)

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

        numpy.savetxt('/root/share/project/didi/data/kitti/dummy/seg/depths.txt',depths)
        numpy.savetxt('/root/share/project/didi/data/kitti/dummy/seg/aspects.txt',aspects)
        numpy.savetxt('/root/share/project/didi/data/kitti/dummy/seg/scales.txt',scales)
        cv2.imwrite('/root/share/project/didi/data/kitti/dummy/seg/top_image/top_rois.png',mean_image)








    #----------------------------------------------------------
    #----------------------------------------------------------
    exit(0)





    #----------------------------------------------------------
    lidar = dataset.velo[0]

    objs = objects[0]
    gt_labels, gt_boxes, gt_boxes3d = obj_to_gt(objs)

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(lidar, fig=fig)
    draw_gt_boxes3d(gt_boxes3d, fig=fig)
    mlab.show(1)

    print ('** calling lidar_to_tops() **')
    if 0:
        top, top_image = lidar_to_top(lidar)
        rgb = dataset.rgb[0][0]
    else:
        top = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/top.npy')
        top_image = cv2.imread('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.png')
        rgb = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy')

    rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # -----------


























    #check
    num = len(gt_boxes)
    for n in range(num):
       x1,y1,x2,y2 = gt_boxes[n]
       cv2.rectangle(top_image,(x1,y1), (x2,y2), (0,255,255), 1)


    ## check
    boxes3d0 = box_to_box3d(gt_boxes)

    draw_gt_boxes3d(boxes3d0,  color=(1,1,0), line_width=1, fig=fig)
    mlab.show(1)

    for n in range(num):
        qs = make_projected_box3d(gt_boxes3d[n])
        draw_projected_box3d(rgb,qs)

    imshow('rgb',rgb)
    cv2.waitKey(0)




    #save
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy',rgb)
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/lidar.npy',lidar)
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/top.npy',top)
    #cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.png',top_image)
    #cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.maked.png',top_image)

    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_labels.npy',gt_labels)
    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes.npy',gt_boxes)
    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes3d.npy',gt_boxes3d)

    imshow('top_image',top_image)
    cv2.waitKey(0)

    pause











    exit(0)





