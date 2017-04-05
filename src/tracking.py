import model as mod
import data
from net.utility.draw import imsave
m3=mod.MV3D()
load_indexs=[ 0,  99, 23, 135]
for i in load_indexs:
    rgbs, tops, fronts, gt_labels, gt_boxes3d=data.load([i])
    boxes3d,probs,img_tracking=m3.tacking(tops[0],fronts[0],rgbs[0])
    imsave('tacking {}'.format(i),img_tracking)