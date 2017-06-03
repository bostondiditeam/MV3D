import os
import numpy as np
# np.random.seed(7)
import tensorflow as tf
# tf.set_random_seed(7)
from sklearn.utils import shuffle
import glob
import net.utility.draw  as nud
import mv3d_net
import net.blocks as blocks
import data
import net.processing.boxes3d  as box
from net.rpn_target_op import make_bases, make_anchors, rpn_target
from net.rcnn_target_op import rcnn_target,fusion_target
from net.rpn_nms_op     import draw_rpn_proposal
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_nms, draw_rcnn,draw_box3d_on_image_with_gt,draw_fusion_target
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels
import net.utility.file as utilfile
from config import cfg
import config
from net.processing.boxes import non_max_suppress
import utils.batch_loading as dataset
from utils.timer import timer
from keras import backend as K
from time import localtime, strftime
import cv2
import time


#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017

def get_top_feature_shape(top_shape, stride):
    return (top_shape[0]//stride, top_shape[1]//stride)

def project_to_roi3d(top_rois):
    num = len(top_rois)
    # rois3d = np.zeros((num,8,3))
    rois3d = box.top_box_to_box3d(top_rois[:, 1:5])
    return rois3d


def project_to_rgb_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    projections = box.box3d_to_rgb_box(rois3d)
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


class MV3D(object):

    def __init__(self, top_shape, front_shape, rgb_shape):

        # anchors
        self.top_stride=None
        self.num_class = 2  # incude background

        ratios=np.array([0.5,1,2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.float32)


        self.bases = make_bases(
            base_size = 16,
            ratios=ratios,  #aspet ratio
            scales=scales
        )

        # output dir, etc
        utilfile.makedirs(cfg.CHECKPOINT_DIR)
        utilfile.makedirs(os.path.join(cfg.CHECKPOINT_DIR,'reg'))
        self.log = utilfile.Logger(cfg.LOG_DIR+'/log.txt', mode='a')
        self.track_log = utilfile.Logger(cfg.LOG_DIR + '/tracking_log.txt', mode='a')


        # about tensorboard.
        self.tb_dir = strftime("%Y_%m_%d_%H_%M", localtime())
        self.train_summary_writer = None
        self.val_summary_writer = tf.summary.FileWriter(os.path.join(cfg.LOG_DIR, 'tensorboard', self.tb_dir + '_val'))
        self.tensorboard_dir = None
        self.summ = None

        self.rpn_checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, 'mv3d_rpn')

        self.fusion_net_checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, 'mv3d_fusion_net')

        self.all_net_checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR)
        os.makedirs(self.rpn_checkpoint_dir, exist_ok=True)
        os.makedirs(self.fusion_net_checkpoint_dir, exist_ok=True)
        os.makedirs(self.all_net_checkpoint_dir, exist_ok=True)

        # creat sesssion
        self.sess = tf.Session()
        self.use_pretrain_weights=[]

        self.build_net(top_shape, front_shape, rgb_shape)


        # set anchor boxes
        self.top_stride = self.net['top_feature_stride']
        top_feature_shape = get_top_feature_shape(top_shape, self.top_stride)
        self.top_view_anchors, self.anchors_inside_inds = make_anchors(self.bases, self.top_stride,                                                                    top_shape[0:2], top_feature_shape[0:2])
        self.anchors_inside_inds = np.arange(0, len(self.top_view_anchors), dtype=np.int32)  # use all  #<todo>

        self.log_subdir = None
        self.top_image = None


        self.batch_top_inds = None
        self.batch_top_labels =  None
        self.batch_top_pos_inds = None
        self.batch_top_targets = None
        self.batch_proposals =None
        self.batch_proposal_scores = None
        self.batch_gt_top_boxes = None
        self.batch_gt_labels = None

        self.time_str = None
        self.frame_info =None


    def predict(self, top_view, front_view, rgb_image):
        self.lables = []  # todo add lables output

        self.top_view = top_view
        self.rgb_image = rgb_image
        self.front_view = front_view
        fd1 = {
            self.net['top_view']: self.top_view,
            self.net['top_anchors']: self.top_view_anchors,
            self.net['top_inside_inds']: self.anchors_inside_inds,
            blocks.IS_TRAIN_PHASE: False,
            K.learning_phase(): True
        }

        self.proposals, self.proposal_scores = \
            self.sess.run([self.net['proposals'], self.net['proposal_scores']], fd1)
        self.proposal_scores = np.reshape(self.proposal_scores, (-1))
        self.top_rois = self.proposals
        if len(self.top_rois) == 0:
            return np.zeros((0, 8, 3)), []

        self.rois3d = project_to_roi3d(self.top_rois)
        self.front_rois = project_to_front_roi(self.rois3d)
        self.rgb_rois = project_to_rgb_roi(self.rois3d)

        fd2 = {
            **fd1,
            self.net['front_view']: self.front_view,
            self.net['rgb_images']: self.rgb_image,

            self.net['top_rois']: self.top_rois,
            self.net['front_rois']: self.front_rois,
            self.net['rgb_rois']: self.rgb_rois,

        }

        self.fuse_probs, self.fuse_deltas = \
            self.sess.run([self.net['fuse_probs'], self.net['fuse_deltas']], fd2)

        self.probs, self.boxes3d = rcnn_nms(self.fuse_probs, self.fuse_deltas, self.rois3d, score_threshold=0.5)

        return self.boxes3d, self.lables

    def predict_log(self, prefix, log_subdir):
        top_image = data.draw_top_image(self.top_view[0])
        self.log_rpn()
        self.log_fusion_net_detail(log_subdir, self.fuse_probs, self.fuse_deltas)
        text_lables = ['No.%d class:1 prob: %.4f' % (i, prob) for i, prob in enumerate(self.probs)]
        img_boxes3d_prediction = nud.draw_box3d_on_camera(self.rgb_image[0], self.boxes3d, text_lables=text_lables)
        new_size = (img_boxes3d_prediction.shape[1] // 2, img_boxes3d_prediction.shape[0] // 2)
        img_boxes3d_prediction = cv2.resize(img_boxes3d_prediction, new_size)
        nud.imsave('%s_img_boxes3d_prediction' % (prefix), img_boxes3d_prediction, log_subdir)



    def batch_data_is_invalid(self,train_gt_boxes3d):
        # todo : support batch size >1

        for i in range(len(train_gt_boxes3d)):
            if box.box3d_in_top_view(train_gt_boxes3d[i]):
                continue
            else:
                return True
        return False


    def get_variables_by_scopes_name(self, scopes):
        variables=[]
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            assert len(variables) != 0
            variables += variables
        return variables


    def get_rpn_variables(self):
        rpn_scopes = ['MV3D/top_view_rpn']
        return self.get_variables_by_scopes_name(rpn_scopes)


    def get_fusion_net_variables(self):
        #fuse-net
        scopes = ['MV3D/fusion']
        return self.get_variables_by_scopes_name(scopes)


    def get_rgb_feature_net_variables(self):
        #'rgb-feature-net'
        return self.get_variables_by_scopes_name(['MV3D/image_feature'])


    def build_net(self, top_shape, front_shape, rgb_shape):
        with tf.variable_scope('MV3D'):
            net = mv3d_net.load(top_shape, front_shape, rgb_shape, self.num_class, len(self.bases))
            self.net = net


    def variables_initializer(self):
        # uninit_vars=[]
        # if 'all' not in self.use_pretrain_weights:
        #     if 'rpn' not in self.use_pretrain_weights:
        #         uninit_vars +=self.get_rpn_variables()
        #
        #     if 'fusion_net' not in self.use_pretrain_weights:
        #         uninit_vars +=self.get_fusion_net_variables()
        #
        #     if config.cfg.USE_IMAGENET_PRE_TRAINED_MODEL == False: # todo : remove it
        #         uninit_vars += self.get_rgb_feature_net_variables()
        #
        # if uninit_vars != []:
        #     self.sess.run(tf.variables_initializer(uninit_vars),
        #              {blocks.IS_TRAIN_PHASE: True, K.learning_phase(): 1})

        # todo : remove it
        self.sess.run(tf.global_variables_initializer(),
                 {blocks.IS_TRAIN_PHASE: True, K.learning_phase(): 1})


    def load_weights(self, weights=[]):
        path=None
        for name in weights:
            if name == 'all':
                path = os.path.join(self.all_net_checkpoint_dir, 'mv3d_all_net.ckpt')
                assert tf.train.checkpoint_exists(path) == True
                print('load all_net pretrained model')
                self.all_net_saver.restore(self.sess, path)
            elif name == 'rpn':
                path = os.path.join(self.rpn_checkpoint_dir, 'rpn.ckpt')
                assert tf.train.checkpoint_exists(path)==True
                print('load rpn pretrained model')
                self.rpn_saver.restore(self.sess, path)

            elif name == 'fusion_net':
                path=os.path.join(self.fusion_net_checkpoint_dir, 'fusion_net.ckpt')
                assert tf.train.checkpoint_exists(path) ==True

                print('load fusion_net pretrained model')
                self.fusion_net_saver.restore(self.sess, path)
            else:
                ValueError('unknow weigths name')


    def save_weights(self, weights=[]):
        path = None
        sess=self.sess
        for name in weights:
            if name == 'all':
                self.all_net_saver.save(sess, os.path.join(self.all_net_checkpoint_dir, 'mv3d_all_net.ckpt'))
                print('all net model save!')

            elif name == 'rpn':
                self.rpn_saver.save(sess, os.path.join(self.rpn_checkpoint_dir, 'rpn.ckpt'))
                print('rpn model save!')

            elif name == 'fusion_net':
                self.fusion_net_saver.save(sess, os.path.join(self.fusion_net_checkpoint_dir, 'fusion_net.ckpt'))
                print('fusion_net model save!')
            else:
                ValueError('unknow weigths name')

    def log_rpn(self):
        top_image = self.top_image
        subdir = self.log_subdir
        top_inds = self.batch_top_inds
        top_labels = self.batch_top_labels
        top_pos_inds = self.batch_top_pos_inds
        top_targets = self.batch_top_targets
        proposals = self.batch_proposals
        proposal_scores = self.batch_proposal_scores
        gt_top_boxes = self.batch_gt_top_boxes
        gt_labels = self.batch_gt_labels

        if gt_top_boxes is not None:
            img_gt = draw_rpn_gt(top_image, gt_top_boxes, gt_labels)
            nud.imsave('img_rpn_gt', img_gt, subdir)

        if top_inds is not None:
            img_label = draw_rpn_labels(top_image, self.top_view_anchors, top_inds, top_labels)
            nud.imsave('img_rpn_label', img_label, subdir)

        if top_pos_inds is not None:
            img_target = draw_rpn_targets(top_image, self.top_view_anchors, top_pos_inds, top_targets)
            nud.imsave('img_rpn_target', img_target, subdir)

        if proposals is not None:
            rpn_proposal = draw_rpn_proposal(top_image, proposals, proposal_scores, draw_num=20)
            nud.imsave('img_rpn_proposal', rpn_proposal, subdir)


    def log_fusion_net_detail(self, subdir, fuse_probs, fuse_deltas):
        dir = os.path.join(cfg.LOG_DIR, subdir)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'fusion_net_detail.txt'), 'w') as info_file:
            info_file.write('index, fuse_probs, fuse_deltas\n')
            for i, prob in enumerate(fuse_probs):
                info_file.write('{}, {}, {}\n'.format(i, prob, fuse_deltas[i]))




class Predictor(MV3D):
    def __init__(self, top_shape, front_shape, rgb_shape):
        MV3D.__init__(self, top_shape, front_shape, rgb_shape)
        self.all_net_saver = tf.train.Saver()
        self.variables_initializer()
        self.load_weights(['all'])


    def __call__(self, top_view, front_view, rgb_image):
        return self.predict(top_view, front_view, rgb_image)




class Trainer(MV3D):

    def __init__(self, train_set, validation_set, pre_trained_weights):
        top_shape, front_shape, rgb_shape = train_set.get_shape()
        MV3D.__init__(self,top_shape, front_shape, rgb_shape)
        self.train_set = train_set
        self.validation_set = validation_set

        # saver
        with self.sess.as_default():

            with tf.variable_scope('minimize_loss'):
                # solver
                # l2 = blocks.l2_regulariser(decay=0.0005)
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                # solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
                solver = tf.train.AdamOptimizer()

                # summary
                self.top_cls_loss = self.net['top_cls_loss']
                tf.summary.scalar('top_cls_loss', self.top_cls_loss)

                self.top_reg_loss = self.net['top_reg_loss']
                tf.summary.scalar('top_reg_loss', self.top_reg_loss)

                self.fuse_cls_loss = self.net['fuse_cls_loss']
                tf.summary.scalar('fuse_cls_loss', self.fuse_cls_loss)

                self.fuse_reg_loss = self.net['fuse_reg_loss']
                tf.summary.scalar('fuse_reg_loss', self.fuse_reg_loss)

                self.trainning_target = 'all'  # 'rpn' 'fusion_net' 'all'
                if self.trainning_target == 'all':
                    total_loss = 1. * (1. * self.top_cls_loss + 0.05 * self.top_reg_loss) + \
                                 1. * self.fuse_cls_loss + 0.1 * self.fuse_reg_loss
                    tf.summary.scalar('total_loss', total_loss)
                    self.solver_step = solver.minimize(total_loss)

                elif self.trainning_target == 'rpn':
                    total_loss = 1. * self.top_cls_loss + 0.05 * self.top_reg_loss
                    tf.summary.scalar('top_total_loss', total_loss)
                    self.solver_step = solver.minimize(total_loss, var_list=self.get_rpn_variables())

                elif self.trainning_target == 'fusion_net':
                    total_loss = 1. * self.fuse_cls_loss + 0.05 * self.fuse_reg_loss
                    tf.summary.scalar('fuse_total_loss', total_loss)
                    self.solver_step = solver.minimize(total_loss, var_list=self.get_fusion_net_variables())

            self.train_summary_writer = tf.summary.FileWriter(os.path.join(cfg.LOG_DIR, 'tensorboard',
                                                                           self.tb_dir + '_train'),
                                                              graph=tf.get_default_graph())
            summ = tf.summary.merge_all()
            self.summ = summ

            self.rpn_saver = tf.train.Saver(self.get_rpn_variables())
            self.fusion_net_saver = tf.train.Saver(self.get_fusion_net_variables())
            self.all_net_saver = tf.train.Saver()
            self.variables_initializer()
            self.load_weights(pre_trained_weights)
            self.n_global_step = 0

    def anchors_details(self):
        pos_indes=self.batch_top_pos_inds
        top_inds=self.batch_top_inds
        return 'anchors: positive= {} total= {}\n'.format(len(pos_indes), len(top_inds))


    def rpn_poposal_details(self):
        top_rois =self.batch_top_rois[0]
        labels =self.batch_fuse_labels[0]
        total = len(top_rois)
        fp = np.sum(labels == 0)
        pos = total - fp
        info = 'RPN proposals: positive= {} total= {}'.format(pos, total)
        return info





    def log_fusion_net_target(self,rgb):
        subdir = self.log_subdir
        top_image = self.top_image
        img_label = draw_rcnn_labels(top_image, self.batch_top_rois, self.batch_fuse_labels)
        img_target = draw_rcnn_targets(top_image, self.batch_top_rois, self.batch_fuse_labels,
                                       self.batch_fuse_targets)
        nud.imsave('img_rcnn_label', img_label, subdir)
        nud.imsave('img_rcnn_target', img_target, subdir)

        img_rgb_rois = box.draw_boxes(rgb, self.batch_rgb_rois[np.where(self.batch_fuse_labels == 0), 1:5][0],
                                      color=(0, 0, 255), thickness=1)
        img_rgb_rois = box.draw_boxes(img_rgb_rois,
                                      self.batch_rgb_rois[np.where(self.batch_fuse_labels == 1), 1:5][0],
                                      color=(255, 255, 255), thickness=3)
        nud.imsave('img_rgb_rois', img_rgb_rois, subdir)

        # labels, deltas, rois3d, top_img, cam_img, class_color
        top_img, cam_img = draw_fusion_target(self.batch_fuse_labels, self.batch_fuse_targets, self.batch_rois3d,
                                              top_image, rgb, [[10, 20, 10], [255, 0, 0]])
        nud.imsave('fusion_target_rgb', cam_img, subdir)
        nud.imsave('fusion_target_top', top_img, subdir)


    def log_prediction(self, batch_top_view, batch_front_view, batch_rgb_images):

        boxes3d, lables = self.predict(batch_top_view, batch_front_view, batch_rgb_images)
        self.predict_log('',self.log_subdir)


    def log_info(self, subdir, info):
        dir = os.path.join(cfg.LOG_DIR, subdir)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'info.txt'), 'w') as info_file:
            info_file.write(info)


    def __call__(self, max_iter=1000, train_set =None, validation_set =None):

        sess = self.sess
        net = self.net

        with sess.as_default():
            #for init model

            iter_debug=200
            batch_size=1

            validation_step=40
            ckpt_save_step=200


            if cfg.TRAINING_TIMER:
                time_it = timer()

            # start training here  #########################################################################################
            self.log.write('iter |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  total |  \n')
            self.log.write('-------------------------------------------------------------------------------------\n')


            for iter in range(1,max_iter):


                is_validation = False
                summary_it = False
                summary_runmeta = False
                print_loss = False
                log_this_iter = False

                # set fit flag
                if iter % validation_step == 0:  summary_it,is_validation,print_loss = True,True,True # summary validation loss
                if (iter+1) % validation_step == 0:  summary_it,print_loss = True,True # summary train loss
                if iter % 5 == 0: print_loss = True #print train loss

                if 0 and  iter%100 == 3: summary_it,summary_runmeta = True,True

                data_set = self.validation_set if is_validation else self.train_set

                step_name = 'validation' if is_validation else 'training'

                # load dataset
                self.batch_rgb_images, self.batch_top_view, self.batch_front_view, \
                self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id = \
                    data_set.load(batch_size, shuffled=True)

                # check
                if self.batch_data_is_invalid(self.batch_gt_boxes3d[0]):
                    continue


                # fit_iterate log init
                if 1 and self.n_global_step % iter_debug == 0:
                    self.time_str = strftime("%Y_%m_%d_%H_%M", localtime())
                    self.frame_info = data_set.get_frame_info(self.frame_id)[0]
                    self.log_subdir = step_name + '/' + self.time_str
                    self.top_image = data.draw_top_image(self.batch_top_view[0])

                    log_this_iter =True


                # fit
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss= \
                    self.fit_iteration(self.batch_rgb_images, self.batch_top_view, self.batch_front_view,
                                       self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id,
                                       is_validation =is_validation, summary_it=summary_it,
                                       summary_runmeta=summary_runmeta, log=log_this_iter)


                if print_loss:
                    self.log.write('%10s: |  %5d  %0.5f   %0.5f   |   %0.5f   %0.5f \n' % \
                                   (step_name, self.n_global_step, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss))


                if iter%ckpt_save_step==0:
                    # saver.save(sess, pretrained_model_path)
                    print('save_weights')
                    self.save_weights([self.trainning_target])


                    if cfg.TRAINING_TIMER:
                        self.log.write('It takes %0.2f secs to train %d iterations. \n' %\
                                       (time_it.time_diff_per_n_loops(), ckpt_save_step))

                self.n_global_step += 1


            if cfg.TRAINING_TIMER:
                self.log.write('It takes %0.2f secs to train the dataset. \n' % \
                               (time_it.total_time()))
        self.train_summary_writer.close()
        self.val_summary_writer.close()



    def fit_iteration(self, batch_rgb_images, batch_top_view, batch_front_view,
                      batch_gt_labels, batch_gt_boxes3d, frame_id, is_validation =False,
                      summary_it=False, summary_runmeta=False, log=False):

        net = self.net
        sess = self.sess

        # put tensorboard inside
        top_cls_loss = net['top_cls_loss']
        top_reg_loss = net['top_reg_loss']
        fuse_cls_loss = net['fuse_cls_loss']
        fuse_reg_loss = net['fuse_reg_loss']


        self.batch_gt_top_boxes = data.box3d_to_top_box(batch_gt_boxes3d[0])

        ## run propsal generation
        fd1 = {
            net['top_view']: batch_top_view,
            net['top_anchors']: self.top_view_anchors,
            net['top_inside_inds']: self.anchors_inside_inds,

            blocks.IS_TRAIN_PHASE: True,
            K.learning_phase(): 1
        }
        self.batch_proposals, self.batch_proposal_scores, self.batch_top_features = \
            sess.run([net['proposals'], net['proposal_scores'], net['top_features']], fd1)

        ## generate  train rois  for RPN
        self.batch_top_inds, self.batch_top_pos_inds, self.batch_top_labels, self.batch_top_targets = \
            rpn_target(self.top_view_anchors, self.anchors_inside_inds, batch_gt_labels[0],
                       self.batch_gt_top_boxes)
        if log: self.log_rpn()


        self.batch_top_rois, self.batch_fuse_labels, self.batch_fuse_targets = \
            fusion_target(self.batch_proposals, batch_gt_labels[0], self.batch_gt_top_boxes, batch_gt_boxes3d[0])

        self.batch_rois3d = project_to_roi3d(self.batch_top_rois)
        self.batch_front_rois = project_to_front_roi(self.batch_rois3d)
        self.batch_rgb_rois = project_to_rgb_roi(self.batch_rois3d)

        if log: self.log_fusion_net_target(batch_rgb_images[0])
        if log:
            log_info_str = 'frame info: ' + self.frame_info + '\n'
            log_info_str += self.anchors_details()
            log_info_str += self.rpn_poposal_details()
            self.log_info(self.log_subdir, log_info_str)

        ## run classification and regression loss -----------
        fd2 = {
            **fd1,

            net['top_view']: batch_top_view,
            net['front_view']: batch_front_view,
            net['rgb_images']: batch_rgb_images,

            net['top_rois']: self.batch_top_rois,
            net['front_rois']: self.batch_front_rois,
            net['rgb_rois']: self.batch_rgb_rois,

            net['top_inds']: self.batch_top_inds,
            net['top_pos_inds']: self.batch_top_pos_inds,
            net['top_labels']: self.batch_top_labels,
            net['top_targets']: self.batch_top_targets,

            net['fuse_labels']: self.batch_fuse_labels,
            net['fuse_targets']: self.batch_fuse_targets,
        }


        if summary_it:
            run_options = None
            run_metadata = None

            if is_validation:
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss, tb_sum_val = \
                    sess.run([top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss, self.summ], fd2)
                self.val_summary_writer.add_summary(tb_sum_val, self.n_global_step)
            else:
                if summary_runmeta:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                _, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss, tb_sum_val = \
                    sess.run([self.solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss,
                              self.summ], feed_dict=fd2, options=run_options, run_metadata=run_metadata)
                self.train_summary_writer.add_summary(tb_sum_val, self.n_global_step)

                if summary_runmeta:
                    self.train_summary_writer.add_run_metadata(run_metadata, 'step%d' % self.n_global_step)
                    print('Add run metadata ')

        else:
            if is_validation:
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                    sess.run([top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss], fd2)
            else:

                _, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                    sess.run([self.solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],
                             feed_dict=fd2)
        if log: self.log_prediction(batch_top_view, batch_front_view, batch_rgb_images)

        return t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss




