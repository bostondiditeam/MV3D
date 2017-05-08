import model as mod
import glob
from config import *
import utils.batch_loading as ub
import cv2
import numpy as np
import net.utility.draw as draw
import skvideo.io
from config import cfg

dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

training_dataset = {
    '1': ['6_f','9_f','15','20', '11', '21_f',],
    '2': ['3_f',],
    '3': ['2_f','4','6','8','7','7', '11_f',],
}

def train_data_render(gt_boxes3d_dir, gt_labels_dir, rgb_dir, top_dir, save_video_name):
    files = glob.glob(os.path.join(gt_boxes3d_dir,"*.npy"))
    files_count = len(files)

    vid_in = skvideo.io.FFmpegWriter(os.path.join(cfg.LOG_DIR, save_video_name))
    for i in range(files_count):
        name = "{:05}".format(i)
        gt_boxes3d_file = os.path.join(gt_boxes3d_dir, name + '.npy')
        gt_labels_file = os.path.join(gt_labels_dir, name + '.npy')
        rgb_file = os.path.join(rgb_dir, name + '.png')
        top_file = os.path.join(top_dir, name + '.npy')
        #print(gt_boxes3d_file)
        boxes3d = np.load(gt_boxes3d_file)
        rgb_image = cv2.imread(rgb_file)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        if len(boxes3d) > 0:
            rgb_image = draw.draw_boxed3d_to_rgb(rgb_image, boxes3d, color=(0, 0, 80), thickness=3)
        rgb_image = cv2.resize(rgb_image, (500, 400))
        vid_in.writeFrame(rgb_image)

    vid_in.close()
    
    #
    #     # for debugging: save image and show image.
    #     top_image = data.draw_top_image(top[0])
    #     rgb_image = rgb[0]
    #     if len(boxes3d)!=0:
    #         top_image = data.draw_box3d_on_top(top_image, boxes3d[:,:,:], color=(80, 80, 0), thickness=3)
    #         rgb_image = draw.draw_boxed3d_to_rgb(rgb_image, boxes3d[:,:,:], color=(0, 0, 80), thickness=3)
    #         translation, size, rotation = boxes3d_for_evaluation(boxes3d[:,:,:])
    #         for j in range(len(translation)):
    #             tracklet.add_tracklet(i, size[j], translation[j], rotation[j])
    #     rgb_image = cv2.resize(rgb_image, (500, 400))
    #     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    #     new_image = np.concatenate((top_image, rgb_image), axis = 1)
    #     imsave('%5d_image'%i, new_image, 'testset')
    #     vid_in.writeFrame(new_image)

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
            output_file = os.path.join(output_dir, "{}__{}.mp4".format(major, minor))
            print("output_file: " + output_file)
            train_data_render(gt_boxes3d_dir, gt_labels_dir, rgb_dir, top_dir, output_file)

    print("Completed")


