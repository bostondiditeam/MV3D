import math
import numpy as np
import cv2
import sys
import net.processing.projection as proj
if sys.version_info[0]>=3:
    from shapely.geometry import Polygon
from config import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
    TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION
from config import cfg
from numba import jit
import config
from config import *


def heat_map_rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return (r, g, b)

##extension for 3d
@jit
def top_to_lidar_coords(xx,yy):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    y = Yn*TOP_Y_DIVISION-(xx+0.5)*TOP_Y_DIVISION + TOP_Y_MIN
    x = Xn*TOP_X_DIVISION-(yy+0.5)*TOP_X_DIVISION + TOP_X_MIN

    return x,y

@jit
def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    xx = Yn-int((y-TOP_Y_MIN)//TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)//TOP_X_DIVISION)

    return xx,yy

@jit
def top_box_to_box3d(boxes):

    num=len(boxes)
    boxes3d = np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        x1,y1,x2,y2 = boxes[n]

        points = [ (x1,y1), (x1,y2), (x2,y2), (x2,y1) ]
        for k in range(4):
            xx,yy = points[k]
            x,y  = top_to_lidar_coords(xx,yy)
            boxes3d[n,k,  :] = x,y, -2  ## <todo>
            boxes3d[n,4+k,:] = x,y,0.4

    return boxes3d

def box3d_in_top_view(boxes3d):
    # what if only some are outside of the range, but majorities are inside.
    for i in range(8):
        if TOP_X_MIN<=boxes3d[i,0]<=TOP_X_MAX and TOP_Y_MIN<=boxes3d[i,1]<=TOP_Y_MAX:
            continue
        else:
            return False
    return True

@jit
def box3d_to_top_box(boxes3d):

    num  = len(boxes3d)
    boxes = np.zeros((num,4),  dtype=np.float32)

    for n in range(num):
        b   = boxes3d[n]

        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)

        umin=min(u0,u1,u2,u3)
        umax=max(u0,u1,u2,u3)
        vmin=min(v0,v1,v2,v3)
        vmax=max(v0,v1,v2,v3)

        boxes[n]=np.array([umin,vmin,umax,vmax])

    return boxes

@jit
def convert_points_to_croped_image(img_points):
    img_points=img_points.copy()

    left=cfg.IMAGE_CROP_LEFT  #pixel
    right=cfg.IMAGE_CROP_RIGHT
    top=cfg.IMAGE_CROP_TOP
    bottom=cfg.IMAGE_CROP_BOTTOM

    croped_img_h=proj.image_height-top-bottom
    croped_img_w=proj.image_width-left-right


    img_points[:,1] -= top
    mask=img_points[:,1] <0
    img_points[mask,1]=0
    out_range_mask =mask

    mask=img_points[:, 1] >= croped_img_h
    img_points[mask, 1]=croped_img_h-1
    out_range_mask=np.logical_or(out_range_mask,mask)

    img_points[:,0] -= left
    mask=img_points[:,0] <0
    img_points[mask,0]=0
    out_range_mask = np.logical_or(out_range_mask, mask)

    mask=img_points[:, 0] >= croped_img_w
    img_points[mask, 0]=croped_img_w-1
    out_range_mask = np.logical_or(out_range_mask, mask)

    return img_points,out_range_mask


@jit
def box3d_to_rgb_box(boxes3d, Mt=None, Kt=None):
    if (cfg.DATA_SETS_TYPE == 'kitti'):
        if Mt is None: Mt = np.array(MATRIX_Mt)
        if Kt is None: Kt = np.array(MATRIX_Kt)

        num  = len(boxes3d)
        projections = np.zeros((num,8,2),  dtype=np.int32)
        for n in range(num):
            box3d = boxes3d[n]
            Ps = np.hstack(( box3d, np.ones((8,1))) )
            Qs = np.matmul(Ps,Mt)
            Qs = Qs[:,0:3]
            qs = np.matmul(Qs,Kt)
            zs = qs[:,2].reshape(8,1)
            qs = (qs/zs)
            projections[n] = qs[:,0:2]
        return projections

    else:
        num = len(boxes3d)
        projections = np.zeros((num, 8, 2), dtype=np.int32)
        for n in range(num):
            box3d=boxes3d[n].copy()
            if np.sum(box3d[:,0]>0) >0:
                box2d = box3d_to_rgb_projection_cv2(box3d)
                box2d,out_range=convert_points_to_croped_image(box2d)
                if np.sum(out_range==False)>=2:
                    projections[n]=box2d
        return projections


@jit
def box3d_to_top_projections(boxes3d):

    num = len(boxes3d)
    projections = np.zeros((num,4,2),  dtype=np.float32)
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)
        projections[n] = np.array([[u0,v0],[u1,v1],[u2,v2],[u3,v3]])

    return projections


def draw_rgb_projections(image, projections, color=(255,0,255), thickness=2, darker=1.0):

    img = (image.copy()*darker).astype(np.uint8)
    num=len(projections)
    for n in range(num):
        qs = projections[n]
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    return img


def draw_box3d_on_top(image, boxes3d,color=(255,255,255), thickness=1,scores=None):

    img = image.copy()
    num =len(boxes3d)
    for n in range(num):
        b   = boxes3d[n]
        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)
        color=heat_map_rgb(0.,1.,scores[n]) if scores is not None else 255
        cv2.line(img, (u0,v0), (u1,v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1,v1), (u2,v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2,v2), (u3,v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3,v3), (u0,v0), color, thickness, cv2.LINE_AA)

    return  img

def draw_boxes(image, boxes, color=(0,255,255), thickness=1, darken=1.0):
    #img = image.copy() * darken
    img = (image.copy()*darken).astype(np.uint8)
    num =len(boxes)
    for n in range(num):
        b = boxes[n]
        cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),color,thickness)

    return img


## regression -------------------------------------------------------
##<todo> refine this normalisation later ... e.g. use log(scale)
def box3d_transform0(et_boxes3d, gt_boxes3d):

    et_centers =   np.sum(et_boxes3d,axis=1, keepdims=True)/8
    et_scales  =   10#*np.sum((et_boxes3d-et_centers)**2, axis=2, keepdims=True)**0.5
    deltas = (et_boxes3d-gt_boxes3d)/et_scales
    return deltas


def box3d_transform_inv0(et_boxes3d, deltas):

    et_centers =  np.sum(et_boxes3d,axis=1, keepdims=True)/8
    et_scales  =  10#*np.sum((et_boxes3d-et_centers)**2, axis=2, keepdims=True)**0.5
    boxes3d = -deltas*et_scales+et_boxes3d

    return boxes3d

@jit
def box3d_transform(et_boxes3d, gt_boxes3d):

    num=len(et_boxes3d)
    deltas=np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        e=et_boxes3d[n]
        center = np.sum(e,axis=0, keepdims=True)/8
        scale = (np.sum((e-center)**2)/8)**0.5

        g=gt_boxes3d[n]
        deltas[n]= (g-e)/scale
    return deltas


@jit
def box3d_transform_inv(et_boxes3d, deltas):

    num=len(et_boxes3d)
    boxes3d=np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        e=et_boxes3d[n]
        center = np.sum(e,axis=0, keepdims=True)/8
        scale = (np.sum((e-center)**2)/8)**0.5

        d=deltas[n]
        boxes3d[n]= e+scale*d

    return boxes3d



def regularise_box3d(boxes3d):

    num = len(boxes3d)
    reg_boxes3d =np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        b=boxes3d[n]

        dis=0
        corners = np.zeros((4,3),dtype=np.float32)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,k+4
            dis +=np.sum((b[i]-b[j])**2) **0.5
            corners[k] = (b[i]+b[j])/2

        dis = dis/4
        b = reg_boxes3d[n]
        for k in range(0,4):
            i,j=k,k+4
            b[i]=corners[k]-dis/2*np.array([0,0,1])
            b[j]=corners[k]+dis/2*np.array([0,0,1])

    return reg_boxes3d


def boxes3d_decompose(boxes3d):

    # translation
    if cfg.DATA_SETS_TYPE == 'didi2' or cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test':
        T_x = np.sum(boxes3d[:, 0:8, 0], 1) / 8.0
        T_y = np.sum(boxes3d[:, 0:8, 1], 1) / 8.0
        T_z = np.sum(boxes3d[:, 0:8, 2], 1) / 8.0
    elif cfg.DATA_SETS_TYPE == 'kitti':
        T_x = np.sum(boxes3d[:, 0:4, 0], 1) / 4.0
        T_y = np.sum(boxes3d[:, 0:4, 1], 1) / 4.0
        T_z = np.sum(boxes3d[:, 0:4, 2], 1) / 4.0

    Points0 = boxes3d[:, 0, 0:2]
    Points1 = boxes3d[:, 1, 0:2]
    Points2 = boxes3d[:, 2, 0:2]

    dis1=np.sum((Points0-Points1)**2,1)**0.5
    dis2=np.sum((Points1-Points2)**2,1)**0.5

    dis1_is_max=dis1>dis2

    #length width heigth
    L=np.maximum(dis1,dis2)
    W=np.minimum(dis1,dis2)
    H=np.sum((boxes3d[:,0,0:3]-boxes3d[:,4,0:3])**2,1)**0.5

    # rotation
    yaw=lambda p1,p2,dis: math.atan2(p2[1]-p1[1],p2[0]-p1[0])
    R_x = np.zeros(len(boxes3d))
    R_y = np.zeros(len(boxes3d))
    R_z = [yaw(Points0[i],Points1[i],dis1[i]) if is_max else  yaw(Points1[i],Points2[i],dis2[i])
           for is_max,i in zip(dis1_is_max,range(len(dis1_is_max)))]
    R_z=np.array(R_z)

    translation = np.c_[T_x,T_y,T_z]
    size = np.c_[H,W,L]
    rotation= np.c_[R_x,R_y,R_z]
    return translation,size,rotation


@jit
def box3d_compose(translation,size,rotation):
    """
    only support compose one box

    """
    h, w, l = size[0],size[1],size[2]
    if cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test':
        h, w = h * 1.1, l
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    elif cfg.DATA_SETS_TYPE == 'kitti':
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
    elif cfg.DATA_SETS_TYPE == 'didi2':
        l, h, w = 1.1 * l, 1.2 * h, 1.1 * w
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    else:
        raise ValueError('unexpected type in cfg.DATA_SETS_TYPE :{}!'.format(cfg.DATA_SETS_TYPE))


        # re-create 3D bounding box in velodyne coordinate system
    yaw = rotation[2]  # other rotations are 0 in all xml files I checked
    # assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
    rotMat = np.array([ \
        [np.cos(yaw), -np.sin(yaw), 0.0], \
        [np.sin(yaw), np.cos(yaw), 0.0], \
        [0.0, 0.0, 1.0]])
    cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T


    box3d = cornerPosInVelo.transpose()

    return box3d



import cv2


def project_point(point,cameraMat,cameraExtrinsicMat,distCoeff):
  cameraXYZ = cameraExtrinsicMat[0:3,0:3].dot(point.transpose()) + cameraExtrinsicMat[0:3, 3]
  x1 = cameraXYZ[0] / cameraXYZ[2]
  y1 = cameraXYZ[1] / cameraXYZ[2]
  r2 = x1 * x1 + y1 * y1
  factor = 1 + distCoeff[0] * r2 + distCoeff[1] * (r2 ** 2) + distCoeff[4] * (r2 ** 3)
  x2 = x1 * factor + 2 * distCoeff[2] * x1 * y1 + distCoeff[3] * (r2 + 2 * x1 * x1)
  y2 = y1 * factor + distCoeff[2] * (r2 + 2 * y1 * y1) + 2 * distCoeff[3] * x1 * y1
  u = cameraMat[0][0] * x2 + cameraMat[0][2]
  v = cameraMat[1][1] * y2 + cameraMat[1][2]
  return [u,v]



@jit
def box3d_to_rgb_projection_cv2(points):
    ##http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    #cameraMatrix=np.array([[1384.621562, 0.000000, 625.888005],
    #                          [0.000000, 1393.652271, 559.626310],
    #                          [0.000000, 0.000000, 1.000000]])

    ##https://github.com/zxf8665905/lidar-camera-calibration/blob/master/Calibration.ipynb
    ##Out[17]:
    #x=np.array([ -1.50231172e-03,  -4.00842946e-01,  -5.30289086e-01,
    #    -2.41054475e+00,   2.41781181e+00,  -2.46716659e+00])
    #tx, ty, tz, rx, ry, rz = x

    #rotVect = np.array([rx, ry, rz])
    #transVect = np.array([tx, ty, tz])

    #distCoeffs=np.array([[-0.152089, 0.270168, 0.003143, -0.005640, 0.000000]])

    #imagePoints, jacobia=cv2.projectPoints(points,rotVect,transVect,cameraMatrix,distCoeffs)
    #imagePoints=np.reshape(imagePoints,(8,2))
    if cfg.OBJ_TYPE == 'car':
        # projMat = np.matrix([[  6.24391515e+02,  -1.35999541e+03,  -3.47685065e+01,  -8.19238784e+02],
        #                  [  5.20528665e+02,   1.80893752e+01,  -1.38839738e+03,  -1.17506110e+03],
        #                  [  9.99547104e-01,   3.36246424e-03,  -2.99045429e-02,  -1.34871685e+00]])
        projMat = np.matrix([[6.22683238e+02,  -1.36093607e+03,  -2.79236972e+01, -7.43021551e+02],
                            [5.23490385e+02,  1.17742119e+01,  -1.38735135e+03, -1.20545539e+03],
                             [9.99611725e-01,   1.94139055e-03,  -2.77962374e-02, -1.29804894e+00]])


    elif cfg.OBJ_TYPE == 'ped':
        projMat = np.matrix([[4.62722387e+02,  -1.42100788e+03,  -8.53563678e+01, -8.47064132e+02],
                             [4.57167576e+02,  -1.17801020e+01,  -1.41059690e+03, -9.51897491e+02],
                             [9.91070422e-01,  -1.09415446e-01,  -7.62081253e-02, -9.41198803e-01]])
    else:
        raise ValueError('Unknown config.OBJ_TYPE: {}'.format(config.OBJ_TYPE))

    imagePoints=[] 
    for pt in points:
        X = projMat*np.matrix(list(pt)+[1]).T
        X = np.array(X[:2,0]/X[2,0]).flatten()
        imagePoints.append(X)
    imagePoints = np.array(imagePoints)

    return imagePoints.astype(np.int)


def box3d_intersection(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0.

    return z_intersection * xy_intersection


def boxes3d_score_iou(gt_boxes3d, pre_boxes3d):
    n_pre_box = pre_boxes3d.shape[0]
    if n_pre_box ==0: return 0.
    n_gt_box = gt_boxes3d.shape[0]

    _, gt_size, _= boxes3d_decompose(gt_boxes3d)
    gt_vol = np.sum(np.prod(gt_size,1))

    _, pre_size, _ = boxes3d_decompose(pre_boxes3d)
    pre_vol = np.sum(np.prod(pre_size,1))

    inters = np.zeros((n_gt_box, n_pre_box))

    for j in range(n_gt_box):
        for i in range(n_pre_box):
            try:
                inters[j, i] = box3d_intersection(gt_boxes3d[j].T, pre_boxes3d[i].T)
            except:
                raise ValueError('Invalid box')

    inter = np.sum(np.max(inters, 1))
    union = gt_vol+ pre_vol -inter

    iou = inter/union
    return iou




if __name__ == '__main__':
    if 0:
        # test boxes3d_for_evaluation
        gt_boxes3d=np.load('gt_boxes3d_135.npy')
        translation, size, rotation =boxes3d_decompose(gt_boxes3d[0])
        print(translation,size,rotation)

    if 1:
        gt_box3d_trans = np.array([
            [1.6,17.5,-1.0],
            [11.6, 17.5, -1.0],
            [21.6, 17.5, -1.0]
        ])
        gt_box3d_size = np.array([
            [1.6, 2.5, 6.0],
            [1.6, 2.5, 6.0],
            [1.6, 2.5, 6.0]
        ])
        gt_box3d_rota = np.array([
            [0., 0., 1.6],
            [0., 0., 1.6],
            [0., 0., 1.6]
        ])

        pre_box3d_trans = np.array([
            [1.6, 17.5, -1.0],
            [11.6, 17.5, -1.0],
            [21.6, 17.5, -1.0]
        ])
        pre_box3d_size = np.array([
            [1.6, 2.5, 6.0],
            [1.6, 2.5, 6.0],
            [1.6, 2.5, 6.0]
        ])
        pre_box3d_rota = np.array([
            [0., 0., 1.6],
            [0., 0., 1.6],
            [0., 0., 1.6]
        ])

        n_box = gt_box3d_trans.shape[0]
        gt_boxes3d=[]
        for i in range(n_box):
            gt_boxes3d.append(box3d_compose(gt_box3d_trans[i], gt_box3d_size[i], gt_box3d_rota[i]))
        gt_boxes3d = np.array(gt_boxes3d)

        n_box = pre_box3d_trans.shape[0]
        pre_boxes3d = []
        for i in range(n_box):
            pre_boxes3d.append(box3d_compose(pre_box3d_trans[i], pre_box3d_size[i], pre_box3d_rota[i]))
        pre_boxes3d = np.array(pre_boxes3d)

        iou= boxes3d_score_iou(gt_boxes3d, pre_boxes3d)
        print('iou = {}'.format(iou))

        iou= boxes3d_score_iou(gt_boxes3d, pre_boxes3d[0:1, :, :])
        print('iou = {}'.format(iou))
