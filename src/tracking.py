import model as mod
import data
m3=mod.MV3D()
rgbs, tops, fronts, gt_labels, gt_boxes3d=data.load([0])

m3.tacking(tops[0],fronts[0],rgbs[0])