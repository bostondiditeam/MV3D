import model as mod
import data
from net.utility.draw import imsave ,draw_boxed3d_to_rgb
from net.processing.boxes3d import boxes3d_for_evaluation
from tracklets import Tracklet_saver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load pre-trained model to make prediction, then save it into '
                                                 ' the predicted results as tracklets')
    parser.add_argument('prediction', type=str, nargs='?', default='tracklet_labels.xml',
                        help='Whether to output result in a tracklet xml file')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default=None,
                        help='Output folder')


    rgbs, tops, fronts, gt_labels, gt_boxes3d=data.load([0])
    m3=mod.MV3D()
    m3.tracking_init(tops[0].shape,fronts[0].shape,rgbs[0].shape)

    load_indexs=[ 135, 0,  99, 23]
    for i in load_indexs:
        rgbs, tops, fronts, gt_labels, gt_boxes3d=data.load([i])
        boxes3d,probs=m3.tacking(tops[0],fronts[0],rgbs[0])
        file_name='tacking_test_img_{}'.format(i)
        img_tracking=draw_boxed3d_to_rgb(rgbs[0],boxes3d)
        imsave(file_name,img_tracking)
        print(file_name+' save ok')

    print('done !')




if __name__ == '__main__':
    # test boxes3d_for_evaluation
    gt_boxes3d=np.load('gt_boxes3d_135.npy')
    translation, size, rotation =boxes3d_for_evaluation(gt_boxes3d[0])
    print(translation,size,rotation)



# a test case
a = Tracklet_saver('./test/')
size = [1,2,3]
transition = [10,20,30]
rotation = [0.1, 0.2, 0.3]
a.add_tracklet(100, size, transition, rotation)
a.add_tracklet(100, size, transition, rotation)
a.write_tracklet()
