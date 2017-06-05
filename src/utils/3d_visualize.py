import os
os.environ["DISPLAY"] = ":0"

# std libs
import glob


# num libs
import math
import random
import numpy as np

import cv2
import mayavi.mlab as mlab
import config





## save mpg:
##    os.system('ffmpeg -y -loglevel 0 -f image2 -r 15 -i %s/test/predictions/%%06d.png -b:v 2500k %s'%(out_dir,out_avi_file))
##
##----------------------------------------------------------------------------

## preset view points
#  azimuth=180,elevation=0,distance=100,focalpoint=[0,0,0]
## mlab.view(azimuth=azimuth,elevation=elevation,distance=distance,focalpoint=focalpoint)
MM_TOP_VIEW  = 180, 0, 120, [0,0,0]
MM_PER_VIEW1 = 120, 30, 70, [0,0,0]
MM_PER_VIEW2 = 30, 45, 100, [0,0,0]
MM_PER_VIEW3 = 120, 30,100, [0,0,0]



## draw  --------------------------------------------


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):

    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def imshow(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    #prs=arr['ring']
    prs = np.clip(prs/15,0,1)

    #draw grid
    if is_grid:
        L=25
        dL=5
        Z=-2
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-L,L+1,dL):
            x1,y1,z1 = -L, y, Z
            x2,y2,z2 =  L, y, Z
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.3,0.3,0.3), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-L,L+1,dL):
            x1,y1,z1 = x,-L, Z
            x2,y2,z2 = x, L, Z
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.3,0.3,0.3), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if is_axis:
        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)

        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, line_width=2, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, line_width=2, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, line_width=2, figure=fig)


    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        #colormap='bone',  #(0.7,0.7,0.7),  #'gnuplot',  #'bone',  #'spectral',  #'copper',
        #color=(0.9,0.9,0.9),
        #color=(0.9,0.9,0),
        scale_factor=1,
        figure=fig)
    # mlab.points3d(
    #     pxs, pys, pzs,
    #     mode='point',  # 'point'  'sphere'
    #     #colormap='bone',  #(0.7,0.7,0.7),  #'gnuplot',  #'bone',  #'spectral',  #'copper',
    #     #color=(0.9,0.9,0.9),
    #     #color=(0.9,0.9,0),
    #     scale_factor=1,
    #     figure=fig)

if __name__ == '__main__':
    maya = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1,1,1))
    data_3d = np.load('./00313.npy')
    draw_didi_lidar(maya, data_3d, is_grid=1, is_axis=1)
    print('yes')
    pass

def draw_didi_boxes3d(fig, boxes3d, is_number=False, color=(1,1,1), line_width=1):

    if boxes3d.shape==(8,3): boxes3d=boxes3d.reshape(1,8,3)

    num = len(boxes3d)
    for n in range(num):
        b = boxes3d[n]

        if is_number:
            mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)




def dir_to_avi(avi_file, png_dir):

    tmp_dir = '~temp_png'
    os.makedirs(tmp_dir, exist_ok=True)

    for i, file in enumerate(sorted(glob.glob(png_dir + '/*.png'))):
        name = os.path.basename(file).replace('.png','')
        ##os.system('cp file %s'%(tmp_dir + '/' + '%06'%i + '.png'))

        png_file = png_dir +'/'+name+'.png'
        tmp_file = tmp_dir + '/%06d.png'%i
        img = cv2.imread(png_file,1)
        draw_shadow_text(img, 'timestamp='+name.replace('_',':'), (5,20),  0.5, (225,225,225), 1)
        imshow('img',img)
        cv2.waitKey(1)
        cv2.imwrite(tmp_file,img)


    os.system('ffmpeg -y -loglevel 0 -f image2 -r 15 -i %s/%%06d.png -b:v 8000k %s'%(tmp_dir,avi_file))
    os.system('rm -rf %s'%tmp_dir)



# run #################################################################

def mark_gt_box3d( lidar_dir, gt_boxes3d_dir, mark_dir,index):

    os.makedirs(mark_dir, exist_ok=True)
    fig   = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(500, 500))

    count=0
    for file in sorted(glob.glob(lidar_dir + '/*.npy')):
        if count != index:
            count += 1
            continue
        count += 1
        name = os.path.basename(file).replace('.npy','')

        lidar_file   = lidar_dir     +'/'+name+'.npy'
        boxes3d_file = gt_boxes3d_dir+'/'+name+'.npy'
        lidar   = np.load(lidar_file)
        boxes3d = np.load(boxes3d_file)

        mlab.clf(fig)
        draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1)
        if len(boxes3d)!=0:
            draw_didi_boxes3d(fig, boxes3d)

        azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
        mlab.view(azimuth,elevation,distance,focalpoint)
        mlab.show()




# # main #################################################################
# # for demo data:  /root/share/project/didi/data/didi/didi-2/Out/1/15
#
# if __name__ == '__main__':
#
#     preprocessed_dir=config.cfg.PREPROCESSED_DATA_SETS_DIR
#     dataset='/1/15'
#     lidar_dir      =preprocessed_dir+ '/lidar'+dataset
#     gt_boxes3d_dir =preprocessed_dir+'/gt_boxes3d'+dataset
#     mark_dir       =config.cfg.LOG_DIR+ '/mark-gt-box3d'
#     avi_file       =config.cfg.LOG_DIR+ '/mark-gt-box3d.avi'
#
#     mark_gt_box3d(lidar_dir,gt_boxes3d_dir,mark_dir,30)
#     # dir_to_avi(avi_file, mark_dir)

