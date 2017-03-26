from net.common import *


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



def box3d_to_rgb_projections(boxes3d, Mt=None, Kt=None):

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


def draw_rgb_projections(image, projections, color=(255,255,255), thickness=2, darker=0.7):

    img = image.copy()*darker
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


def draw_box3d_on_top(image, boxes3d,color=(255,255,255), thickness=1, darken=0.7):

    img = image.copy()*darken
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

def draw_boxes(image, boxes, color=(0,255,255), thickness=1, darken=0.7):

    img = image.copy()*darken
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


##<todo> refine this regularisation later
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