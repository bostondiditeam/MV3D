import model as mod
import data
from net.utility.draw import imsave ,draw_boxed3d_to_rgb

if __name__ == '__main__':

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