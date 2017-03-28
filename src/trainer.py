# from net.common import *
# from net.utility.file import *
# from net.processing.boxes import *
# from net.processing.boxes3d import *
import net.utility.draw  as nud

from dummynet import *
from data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target
from net.rcnn_target_op import rcnn_target

from net.rpn_nms_op     import draw_rpn_nms, draw_rpn
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_nms, draw_rcnn
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels

dummy_data_dir='../data/kitti/dummy/'

#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017

def load_dummy_data():
    rgb   = np.load(dummy_data_dir + 'one_frame/rgb.npy')
    lidar = np.load(dummy_data_dir + 'one_frame/lidar.npy')
    top   = np.load(dummy_data_dir + 'one_frame/top.npy')
    front = np.zeros((1,1),dtype=np.float32)
    gt_labels    = np.load(dummy_data_dir + 'one_frame/gt_labels.npy')
    gt_boxes3d   = np.load(dummy_data_dir + 'one_frame/gt_boxes3d.npy')
    gt_top_boxes = np.load(dummy_data_dir + 'one_frame/gt_top_boxes.npy')

    top_image   = cv2.imread(dummy_data_dir + 'one_frame/top_image.png')
    front_image = np.zeros((1,1,3),dtype=np.float32)

    rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_boxes3d = gt_boxes3d.reshape(-1,8,3)

    return  rgb, top, front, gt_labels, gt_boxes3d, top_image, front_image, lidar



def load_dummy_datas():

    # doto : num_frames modify
    # num_frames = 154
    num_frames =1
    rgbs      =[]
    lidars    =[]
    tops      =[]
    fronts    =[]
    gt_labels =[]
    gt_boxes3d=[]

    top_images  =[]
    front_images=[]

    # fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    for n in range(num_frames):
        print(n)

        rgb   = cv2.imread(dummy_data_dir + 'seg/rgb/rgb_%05d.png'%n,1)
        lidar = np.load(dummy_data_dir + 'seg/lidar/lidar_%05d.npy'%n)
        top   = np.load(dummy_data_dir + 'seg/top/top_%05d.npy'%n)
        front = np.zeros((1,1),dtype=np.float32)
        gt_label  = np.load(dummy_data_dir + 'seg/gt_labels/gt_labels_%05d.npy'%n)
        gt_box3d = np.load(dummy_data_dir + 'seg/gt_boxes3d/gt_boxes3d_%05d.npy'%n)


        top_image   = cv2.imread(dummy_data_dir + 'seg/top_image/top_image_%05d.png'%n,1)
        front_image = np.zeros((1,1,3),dtype=np.float32)

        rgbs.append(rgb)
        lidars.append(lidar)
        tops.append(top)
        fronts.append(front)
        gt_labels.append(gt_label)
        gt_boxes3d.append(gt_box3d)

        top_images.append(top_image)
        front_images.append(front_image)


        # explore dataset:

        print(gt_box3d)
        if 0:
            projections=box3d_to_rgb_projections(gt_box3d)
            rgb1 = draw_rgb_projections(rgb, projections, color=(255,255,255), thickness=2)
            top_image1 = draw_box3d_on_top(top_image, gt_box3d, color=(255,255,255), thickness=2)

            nud.imsave('rgb', rgb1)
            nud.imsave('top_image', top_image1)

            # mlab.clf(fig)
            # draw_lidar(lidar, fig=fig)
            # draw_gt_boxes3d(gt_box3d, fig=fig)
            # mlab.show(1)
            cv2.waitKey(1)

            pass


    ##exit(0)
    # mlab.close(all=True)
    return  rgbs, tops, fronts, gt_labels, gt_boxes3d, top_images, front_images, lidars



#<todo>
def project_to_roi3d(top_rois):
    num = len(top_rois)
    rois3d = np.zeros((num,8,3))
    rois3d = top_box_to_box3d(top_rois[:,1:5])
    return rois3d


def project_to_rgb_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    projections = box3d_to_rgb_projections(rois3d)
    for n in range(num):
        qs = projections[n]
        minx = np.min(qs[:,0])
        maxx = np.max(qs[:,0])
        miny = np.min(qs[:,1])
        maxy = np.max(qs[:,1])
        rois[n,1:5] = minx,miny,maxx,maxy

    return rois


# todo: finished project_to_front_roi
def  project_to_front_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)

    return rois


def run_train():

    # output dir, etc
    out_dir = '../out/didi/xxx'
    makedirs(out_dir +'/tf')
    makedirs(out_dir +'/check_points')
    log = Logger(out_dir+'/log.txt',mode='a')

    # lidar data -----------------
    if 1:
        ratios=np.array([0.5,1,2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.float32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,  #aspect ratio
            scales=scales
        )
        num_bases = len(bases)
        stride = 8

        rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, lidars = load_dummy_datas()
        num_frames = len(rgbs)

        top_shape   = tops[0].shape
        front_shape = fronts[0].shape
        rgb_shape   = rgbs[0].shape
        top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)
        out_shape=(8,3)


        #-----------------------
        #check data
        # if 0:
        #     fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
        #     draw_lidar(lidars[0], fig=fig)
        #     draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
        #     mlab.show(1)
        #     cv2.waitKey(1)



    # set anchor boxes
    num_class = 2 #incude background
    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all  #<todo>
    print('out_shape=%s'%str(out_shape))
    print('num_frames=%d'%num_frames)


    # load model ####################################################################################################
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_images   = tf.placeholder(shape=[None, *top_shape  ], dtype=tf.float32, name='top'  )
    front_images = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images   = tf.placeholder(shape=[None, *rgb_shape  ], dtype=tf.float32, name='rgb'  )
    top_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'   ) #<todo> change to int32???
    front_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='front_rois' )
    rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )

    top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

    front_features = front_feature_net(front_images)
    rgb_features   = rgb_feature_net(rgb_images)

    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
			( [top_features,     top_rois,     6,6,1./stride],
			  [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
			  [rgb_features,     rgb_rois,     6,6,1./stride],),
            num_class, out_shape) #<todo>  add non max suppression



    #loss ########################################################################################################
    top_inds     = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_ind'    )
    top_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_pos_ind')
    top_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_label'  )
    top_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target' )
    top_cls_loss, top_reg_loss = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    fuse_cls_loss, fuse_reg_loss = rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)


    # solver
    l2 = l2_regulariser(decay=0.0005)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #solver_step = solver.minimize(top_cls_loss+top_reg_loss+l2)
    solver_step = solver.minimize(top_cls_loss+top_reg_loss+fuse_cls_loss+0.1*fuse_reg_loss+l2)

    max_iter = 10000
    iter_debug=8

    # start training here  #########################################################################################
    log.write('epoch     iter    rate   |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  |  \n')
    log.write('-------------------------------------------------------------------------------------\n')

    num_ratios=len(ratios)
    num_scales=len(scales)
    # fig, axs = plt.subplots(num_ratios,num_scales)

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        for iter in range(max_iter):
            epoch=1.0*iter
            rate=0.05


            ## generate train image -------------
            idx = np.random.choice(num_frames)     #*10   #num_frames)  #0
            batch_top_images    = tops[idx].reshape(1,*top_shape)
            batch_front_images  = fronts[idx].reshape(1,*front_shape)
            batch_rgb_images    = rgbs[idx].reshape(1,*rgb_shape)

            batch_gt_labels    = gt_labels[idx]
            batch_gt_boxes3d   = gt_boxes3d[idx]
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)


            ## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds,

                learning_rate:   rate,
                IS_TRAIN_PHASE:  True
            }
            batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)

            ## generate  train rois  ------------
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets  = \
                rpn_target ( anchors, inside_inds, batch_gt_labels,  batch_gt_top_boxes)

            batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                 rcnn_target(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )

            batch_rois3d	 = project_to_roi3d    (batch_top_rois)
            batch_front_rois = project_to_front_roi(batch_rois3d  )
            batch_rgb_rois   = project_to_rgb_roi  (batch_rois3d  )


            ##debug gt generation
            if 1 and iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]

                img_gt     = draw_rpn_gt(top_image, batch_gt_top_boxes, batch_gt_labels)
                img_label  = draw_rpn_labels (top_image, anchors, batch_top_inds, batch_top_labels )
                img_target = draw_rpn_targets(top_image, anchors, batch_top_pos_inds, batch_top_targets)
                nud.imsave('img_rpn_gt', img_gt)
                nud.imsave('img_rpn_label', img_label)
                nud.imsave('img_rpn_target', img_target)

                img_label  = draw_rcnn_labels (top_image, batch_top_rois, batch_fuse_labels )
                img_target = draw_rcnn_targets(top_image, batch_top_rois, batch_fuse_labels, batch_fuse_targets)
                nud.imsave('img_rcnn_label', img_label)
                nud.imsave('img_rcnn_target', img_target)


                img_rgb_rois = draw_boxes(rgb, batch_rgb_rois[:,1:5], color=(255,0,255), thickness=1)
                nud.imsave('img_rgb_rois', img_rgb_rois)

                # cv2.waitKey(1)

            ## run classification and regression loss -----------
            fd2={
				**fd1,

                top_images: batch_top_images,
                front_images: batch_front_images,
                rgb_images: batch_rgb_images,

				top_rois:   batch_top_rois,
                front_rois: batch_front_rois,
                rgb_rois:   batch_rgb_rois,

                top_inds:     batch_top_inds,
                top_pos_inds: batch_top_pos_inds,
                top_labels:   batch_top_labels,
                top_targets:  batch_top_targets,

                fuse_labels:  batch_fuse_labels,
                fuse_targets: batch_fuse_targets,
            }
            #_, batch_top_cls_loss, batch_top_reg_loss = sess.run([solver_step, top_cls_loss, top_reg_loss],fd2)


            _, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss = \
               sess.run([solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],fd2)

            log.write('%3.1f   %d   %0.4f   |   %0.5f   %0.5f   |   %0.5f   %0.5f  \n' %\
				(epoch, iter, rate, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss))



            #debug: ------------------------------------

            if iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]

                batch_top_probs, batch_top_scores, batch_top_deltas  = \
                    sess.run([ top_probs, top_scores, top_deltas ],fd2)

                batch_fuse_probs, batch_fuse_deltas = \
                    sess.run([ fuse_probs, fuse_deltas ],fd2)

                #batch_fuse_deltas=0*batch_fuse_deltas #disable 3d box prediction
                probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.5)


                ## show rpn score maps
                p = batch_top_probs.reshape( *(top_feature_shape[0:2]), 2*num_bases)
                # for n in range(num_bases):
                #     r=n%num_scales
                #     s=n//num_scales
                #     pn = p[:,:,2*n+1]*255
                #     axs[s,r].cla()
                #     axs[s,r].imshow(pn, cmap='gray', vmin=0, vmax=255)
                # plt.pause(0.01)

                # show rpn(top) nms
                img_rpn     = draw_rpn(top_image, batch_top_probs, batch_top_deltas, anchors, inside_inds)
                img_rpn_nms = draw_rpn_nms(top_image, batch_proposals, batch_proposal_scores)
                nud.imsave('img_rpn', img_rpn)
                nud.imsave('img_rpn_nms', img_rpn_nms)
                # cv2.waitKey(1)

                ## show rcnn(fuse) nms
                img_rcnn     = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d,darker=1)
                img_rcnn_nms = draw_rcnn_nms(rgb, boxes3d, probs)
                nud.imsave('img_rcnn', img_rcnn)
                nud.imsave('img_rcnn_nms', img_rcnn_nms)
                # cv2.waitKey(1)

            # save: ------------------------------------
            if iter%500==0:
                #saver.save(sess, out_dir + '/check_points/%06d.ckpt'%iter)  #iter
                saver.save(sess, out_dir + '/check_points/snap.ckpt')  #iter






## main function ##########################################################################

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_train()