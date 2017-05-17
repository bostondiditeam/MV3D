from net.common import *
import math
import numpy as np
import cv2
import net.processing.projection as proj

##extension for 3d
def top_to_lidar_coords(xx,yy):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    y = Xn*TOP_Y_DIVISION-(xx+0.5)*TOP_Y_DIVISION + TOP_Y_MIN
    x = Yn*TOP_X_DIVISION-(yy+0.5)*TOP_X_DIVISION + TOP_X_MIN

    return x,y


def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    xx = Yn-int((y-TOP_Y_MIN)//TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)//TOP_X_DIVISION)

    return xx,yy


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
    for i in range(8):
        if TOP_X_MIN<=boxes3d[i,0]<=TOP_X_MAX and TOP_Y_MIN<=boxes3d[i,1]<=TOP_Y_MAX:
            continue
        else:
            return False
    return True

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



def box3d_to_rgb_projections(boxes3d, Mt=None, Kt=None):
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


def draw_box3d_on_top(image, boxes3d,color=(255,255,255), thickness=1):

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

def box3d_to_rgb_projection_cv2(points):
    #http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    cameraMatrix=np.array([[1384.621562, 0.000000, 625.888005],
                              [0.000000, 1393.652271, 559.626310],
                              [0.000000, 0.000000, 1.000000]])

    #https://github.com/zxf8665905/lidar-camera-calibration/blob/master/Calibration.ipynb
    #Out[17]:
    x=np.array([ -1.50231172e-03,  -4.00842946e-01,  -5.30289086e-01,
        -2.41054475e+00,   2.41781181e+00,  -2.46716659e+00])
    tx, ty, tz, rx, ry, rz = x

    rotVect = np.array([rx, ry, rz])
    transVect = np.array([tx, ty, tz])

    distCoeffs=np.array([[-0.152089, 0.270168, 0.003143, -0.005640, 0.000000]])

    imagePoints, jacobia=cv2.projectPoints(points,rotVect,transVect,cameraMatrix,distCoeffs)
    imagePoints=np.reshape(imagePoints,(8,2))
    return imagePoints.astype(np.int)

if __name__ == '__main__':
    # test boxes3d_for_evaluation
    gt_boxes3d=np.load('gt_boxes3d_135.npy')
    translation, size, rotation =boxes3d_decompose(gt_boxes3d[0])
    print(translation,size,rotation)
