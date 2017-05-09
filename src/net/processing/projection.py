import numpy as np
import math
from config import cfg

image_height=cfg.IMAGE_HEIGHT
image_width=cfg.IMAGE_WIDTH

# ====
P = np.array([[1362.184692, 0.0, 620.575531], [0.0, 1372.305786, 561.873133], [0.0, 0.0, 1.0]])
# P = np.array([[1384.621562, 0.0, 625.888005], [0.0, 1393.652271, 559.626310], [0.0, 0.0, 1.0]])

ry=5.2/180.0*math.pi
ry_M=np.array([[math.cos(ry), 0., math.sin(ry)], [0.0, 1.0, 0.], [-math.sin(ry), 0.0, math.cos(ry)]])

rz=-1.2/180.0*math.pi
rz_M=np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0.], [0, 0.0, 1]])

R_axis = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
cameraMatrix_in = np.array([[1384.621562, 0.000000, 625.888005],
                            [0.000000, 1393.652271, 559.626310],
                            [0.000000, 0.000000, 1.000000]])

# return n X 3
def distortion_correct(points):
    kc = [-0.152089, 0.270168, 0.003143, -0.005640, 0.000000]
    # normal
    n_points_x = points[:, 0]/points[:, 2]
    n_points_y = points[:, 1]/points[:, 2]
    n_points = []
    for i in range(len(n_points_x)):
        x = n_points_x[i]
        y = n_points_y[i]
        r = math.sqrt(x**2 + y**2)
        coeff1 = 1 + kc[0]*(r**2) + kc[1]*(r**4) + kc[4]*(r**6)
        d_x = 2*kc[2]*x*y + kc[3]*(r**2 + 2*(x**2))
        d_y = kc[2]*(r**2+2*(y**2)) + 2*kc[3]*x*y
        i_x = coeff1*x + d_x
        i_y = coeff1*y + d_y
        n_points.append([i_x, i_y, 1])
    return n_points


# return n X 2
def project_cam(points):
    p_tmp=np.dot(ry_M,points.T)
    p_tmp = np.dot(rz_M,p_tmp)
    p_tmp = np.dot(R_axis, p_tmp)
    points = distortion_correct(p_tmp.T)
    p_tmp = np.array(points)
    p_cam = np.dot(P, p_tmp.T)
    p_cam[0, :] = p_cam[0, :] / p_cam[2, :]
    p_cam[1, :] = p_cam[1, :] / p_cam[2, :]
    p_col = p_cam[0, :]
    mask_col = p_col > 0
    p_cam = p_cam[:, mask_col]
    p_col = p_col[mask_col]
    mask_col = p_col < image_width
    p_col = p_col[mask_col]
    p_cam = p_cam[:, mask_col]
    p_row = p_cam[1, :]
    mask_row = p_row > 0
    p_cam = p_cam[:, mask_row]
    p_row = p_row[mask_row]
    mask_row = p_row < image_height
    p_cam = p_cam[:, mask_row]
    pixels_cam = p_cam.T
    if len(pixels_cam)!=8:
        return np.zeros((8,2))


    pixels_cam[:, 1] = image_height - pixels_cam[:, 1]  #- 423

    # pixels_cam[:, 0] = pixels_cam[:, 0] * 1242.0/1368.0
    # pixels_cam[:, 1] = pixels_cam[:, 1] * 375.0/413.0

    pixels_cam = [[int(round(p[0])), int(round(p[1]))] for p in pixels_cam]
    pixels_cam = np.array(pixels_cam)
    # pixels_cam[:, 1] = pixels_cam[:, 1] - 260
    # print pixels_cam
    return pixels_cam


# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                        POINT_CLOUD_TO_PANORAMA
# ==============================================================================
def point_cloud_to_panorama(points,
                            v_res=0.42,
                            h_res = 0.35,
                            v_fov = (-24.9, 2.0),
                            d_range = (0,100),
                            y_fudge=3
                            ):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.

    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    #d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, d_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total/v_res) / (v_fov_total* (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0]* (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below+h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
    img[y_img, x_img] = scale_to_255(d_points, min=d_range[0], max=d_range[1])

    return img