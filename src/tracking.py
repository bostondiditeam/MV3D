import model as mod
import data
from net.utility.draw import imsave ,draw_boxed3d_to_rgb
from net.processing.boxes3d import boxes3d_for_evaluation
from tracklets.Tracklet_saver import Tracklet_saver
import argparse
import os
from config import cfg
import time

# Set true if you want score after export predicted tracklet xml
# set false if you just want to export tracklet xml

def pred_and_save(tracklet_pred_dir, dates, drivers, frames_index=None):
    # Tracklet_saver will check whether the file already exists.
    tracklet = Tracklet_saver(tracklet_pred_dir)

    data_seg = cfg.PREPROCESSED_DATA_SETS_DIR

    file_names_list = sorted(data.get_all_file_names(data_seg, dates, drivers))
    if frames_index != None:
        file_names_list = [file_names_list[i] for i in frames_index]

    rgbs, tops, fronts, _, _ = data.load([file_names_list[0]],is_testset=True)
    m3=mod.MV3D()
    m3.tracking_init(tops[0].shape,fronts[0].shape,rgbs[0].shape)

    for i in range(len(file_names_list)):
        rgb, top, front, _, _ = data.load([file_names_list[i]], is_testset=True)
        t1=time.time()
        boxes3d,probs=m3.tacking(top[0],front[0],rgb[0])
        t2=time.time()
        print('time= '+ str(t2-t1))

        # for debugging: save image and show image.
        top_image=data.getTopImages([file_names_list[i]])[0]
        if len(boxes3d)!=0:
            top_image = data.draw_box3d_on_top(top_image, boxes3d[0:1,:,:], color=(0, 0, 80))
            translation, size, rotation = boxes3d_for_evaluation(boxes3d[0:1,:,:])
            for j in range(len(translation)):
                tracklet.add_tracklet(i, size[j]*2, translation[j], rotation[j])
        imsave(file_names_list[i],top_image)

        # file_name='tacking_test_img_{}'.format(i)
        # img_tracking=draw_boxed3d_to_rgb(rgbs[0],boxes3d)
        # path=os.path.join(cfg.LOG_DIR,file_name)
        # imsave(path,img_tracking)
        # print(path+' save ok')

        # save boxes3d as tracklet files.


    tracklet.write_tracklet()
    print("tracklet file named tracklet_labels.xml is written successfully.")
    return tracklet.path


from tracklets.evaluate_tracklets import tracklet_score

if __name__ == '__main__':
    tracklet_pred_dir = cfg.PREDICTED_XML_DIR

    dates = ['Round1Test']
    drivers = ['19_f2']
    frames_index = None
    # generate tracklet file
    pred_file = pred_and_save(tracklet_pred_dir, dates,drivers,frames_index)
    if_score = False

    if(if_score):
        # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
        #  fits you needs.
        gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, '2011_09_26', '2011_09_26_drive_0005_sync', 'tracklet_labels.xml')
        tracklet_score(pred_file=pred_file, gt_file=gt_tracklet_file, output_dir=tracklet_pred_dir)
        print("scores are save under {} directory.".format(tracklet_pred_dir))

    print("Completed")