import mv3d as mod
import glob
from config import *
import utils.batch_loading as ub
import cv2
import numpy as np
import net.utility.draw as draw
import skvideo.io
import data
from config import cfg
from config import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
    TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION

dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

training_dataset = {
    '1': ['6_f','9_f','15','20', '11', '21_f',],
    '2': ['3_f',],
    '3': ['2_f','4','6','8','7', '11_f',],
}

# training_dataset = {
#     '1': ['6_f',],
#     #'3': ['2_f'],
# }

cameraMatrix = np.array([[1384.621562, 0.000000, 625.888005],
                            [0.000000, 1393.652271, 559.626310],
                            [0.000000, 0.000000, 1.000000]])

cameraDist = np.array([-0.152089, 0.270168, 0.003143, -0.005640, 0.000000])

def filter_center_car(lidar):
    lidar = lidar[np.logical_or(np.abs(lidar[:, 0]) > 4.7/2, np.abs(lidar[:, 1]) > 2.1/2)]
    return lidar


## lidar to top ##
def lidar_to_top(lidar):
    lidar = lidar[lidar[:,0]>TOP_X_MIN]
    lidar = lidar[lidar[:,0]<TOP_X_MAX]
    lidar = lidar[lidar[:,1]>TOP_Y_MIN]
    lidar = lidar[lidar[:,1]<TOP_Y_MAX]
    lidar = lidar[lidar[:,2]>-4]
    lidar = lidar[lidar[:,2]<1]
    lidar=filter_center_car(lidar)

    quantized = (lidar - [TOP_X_MIN, TOP_Y_MIN, -8, 0]) / [TOP_X_DIVISION, TOP_Y_DIVISION, 1., 1]

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    #Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Xn - X0
    width   = Yn - Y0
    #channel = Zn - Z0  + 2
    grid = [[[] for y in range(width)] for x in range(height)]
    top = np.zeros(shape=(height, width), dtype=np.float32)

    for i in range(len(quantized)) :
        grid[int(quantized[i][0])][int(quantized[i][1])].append(quantized[i][2])

    for x in range(height):
        for y in range(width):
            if len(grid[x][y]) > 0:
                top[height-x-1][width-y-1] = max(grid[x][y])
    return top


## lidar to top ##
def lidar_to_front(lidar):
    lidar = lidar[lidar[:,0]>0]
    lidar = lidar[lidar[:,0]<100]
    lidar = lidar[lidar[:,1]>TOP_Y_MIN]
    lidar = lidar[lidar[:,1]<TOP_Y_MAX]
    lidar = lidar[lidar[:,2]>-4.]
    lidar = lidar[lidar[:,2]<1.]
    lidar=filter_center_car(lidar)

    y_division = 0.1
    z_division = 0.1

    quantized = (lidar - [0, TOP_Y_MIN, -4., 0]) / [1, y_division,z_division, 1]

    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//y_division)+1
    Z0, Zn = 0, int((1.-(-4.))//z_division)+1
    height  = Zn - Z0
    width   = Yn - Y0
    grid = [[[] for y in range(width)] for z in range(height)]
    front = np.zeros(shape=(height, width), dtype=np.float32)

    #print("height:{} - width:{}".format(height, width))

    for i in range(len(quantized)):
        grid[int(quantized[i][2])][int(quantized[i][1])].append(quantized[i][0])

    for z in range(height):
        for y in range(width):
            if len(grid[z][y]) > 0:
                #print("z:{} - y:{}".format(z, y))
                front[height-z-1][width-y-1] = min(grid[z][y])
    return front

def draw_top_image(top):
    top_binary = np.zeros_like(top)
    top_binary[top > 0] = 255
    return np.dstack((top_binary, top_binary, top_binary)).astype(np.uint8)
    #return top_image

def draw_front_image(top):
    top_binary = np.zeros_like(top)
    top_binary[top > 0] = 255
    return np.dstack((top_binary, top_binary, top_binary)).astype(np.uint8)


def train_data_render(gt_boxes3d_dir, gt_labels_dir, rgb_dir, top_dir, lidar_dir, save_video_name):
    files = glob.glob(os.path.join(gt_boxes3d_dir,"*.npy"))
    files_count = len(files)

    vid_in = skvideo.io.FFmpegWriter(os.path.join(cfg.LOG_DIR, save_video_name))
    for i in range(files_count):
        name = "{:05}".format(i)
        gt_boxes3d_file = os.path.join(gt_boxes3d_dir, name + '.npy')
        gt_labels_file = os.path.join(gt_labels_dir, name + '.npy')
        rgb_file = os.path.join(rgb_dir, name + '.png')
        top_file = os.path.join(top_dir, name + '.npy')
        lidar_file = os.path.join(lidar_dir, name + ".npy")
        #print(gt_boxes3d_file)
        boxes3d = np.load(gt_boxes3d_file)
        rgb_image = cv2.imread(rgb_file)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        rgb_image_undistort = cv2.undistort(rgb_image, cameraMatrix, cameraDist, None, cameraMatrix)
        #top = np.load(top_file)
        #top_image = data.draw_top_image(top)
        lidar = np.load(lidar_file)
        top = lidar_to_top(lidar)
        top_image = draw_top_image(top)
        front = lidar_to_front(lidar)
        front_image = draw_front_image(front)

        lidar_to_top(lidar)
        if len(boxes3d) > 0:
            rgb_image = draw.draw_box3d_on_camera(rgb_image, boxes3d, color=(0, 0, 255), thickness=1)
            #rgb_image_undistort = draw.draw_boxed3d_to_rgb(rgb_image_undistort, boxes3d, color=(0, 0, 80), thickness=3)
            top_image_boxed = data.draw_box3d_on_top(top_image, boxes3d[:, :, :], color=(255, 255, 0), thickness=1)

        #rgb_image_undistort = cv2.resize(rgb_image_undistort, (500, 400))
        new_image = np.concatenate((top_image, top_image_boxed), axis=1)
        new_image = np.concatenate((front_image, new_image), axis=0)
        rgb_image = cv2.resize(rgb_image, (int(new_image.shape[0] * rgb_image.shape[1] / rgb_image.shape[0]), new_image.shape[0]))
        new_image = np.concatenate((new_image, rgb_image), axis=1)
        vid_in.writeFrame(new_image)

    vid_in.close()

    pass

if __name__ == '__main__':
    data_dir = cfg.PREPROCESSED_DATA_SETS_DIR
    output_dir = cfg.LOG_DIR
    frames_index = None

    dataset_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, {'Round1Test': ['19_f2'],},is_testset=True)

    for major,minors in training_dataset.items():
        for minor in minors:
            gt_boxes3d_dir = os.path.join(data_dir, "gt_boxes3d", major, minor)
            gt_labels_dir = os.path.join(data_dir, "gt_lables", major, minor)
            rgb_dir = os.path.join(data_dir, "rgb", major, minor)
            top_dir = os.path.join(data_dir, "top", major, minor)
            lidar_dir = os.path.join(data_dir, "lidar", major, minor)
            output_file = os.path.join(output_dir, "{}__{}.mp4".format(major, minor))
            print("output_file: " + output_file)
            train_data_render(gt_boxes3d_dir, gt_labels_dir, rgb_dir, top_dir, lidar_dir, output_file)

    print("Completed")


