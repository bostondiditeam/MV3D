from net.lib.utils.bbox import bbox_overlaps as box_overlaps
import numpy as np

if __name__ == '__main__':
    bbox=np.array( [[0.,0.,0.,1.,1.]])
    bbox_gt=np.array( [[0.,0.5,0.5,1.5,1.5]])
    overlaps=box_overlaps(bbox,bbox_gt)
    print(overlaps)
