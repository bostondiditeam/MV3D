import os
import numpy as np
# np.random.seed(7)
import tensorflow as tf
# tf.set_random_seed(7)
from sklearn.utils import shuffle
import glob
from config import cfg
import config
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

from net.processing.boxes import non_max_suppress
import utils.batch_loading as dataset
from utils.timer import timer
from keras import backend as K
from time import localtime, strftime
import cv2
import time
import io
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
import pickle
import subprocess
import sys


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




class Net(object):

    def __init__(self, prefix, scope_name, checkpoint_dir=None):
        self.name =scope_name
        self.prefix = prefix
        self.checkpoint_dir =checkpoint_dir
        self.subnet_checkpoint_dir = os.path.join(checkpoint_dir, scope_name)
        self.subnet_checkpoint_name = scope_name
        os.makedirs(self.subnet_checkpoint_dir, exist_ok=True)
        self.variables = self.get_variables([prefix+'/'+scope_name])
        self.saver=  tf.train.Saver(self.variables)


    def save_weights(self, sess=None, dir=None):
        path = os.path.join(dir, self.subnet_checkpoint_name)
        print('\nSave weigths : %s' % path)
        os.makedirs(dir,exist_ok=True)
        self.saver.save(sess, path)

    def clean_weights(self):
        command = 'rm -rf %s' % (os.path.join(self.subnet_checkpoint_dir))
        subprocess.call(command, shell=True)
        print('\nClean weights: %s' % command)
        os.makedirs(self.subnet_checkpoint_dir ,exist_ok=True)


    def load_weights(self, sess=None):
        path = os.path.join(self.subnet_checkpoint_dir, self.subnet_checkpoint_name)
        if tf.train.checkpoint_exists(path) ==False:
            print('\nCan not found :\n"%s",\nuse default weights instead it\n' % (path))
            path = path.replace(os.path.basename(self.checkpoint_dir),'default')
        assert tf.train.checkpoint_exists(path) == True
        self.saver.restore(sess, path)


    def get_variables(self, scope_names):
        variables=[]
        for scope in scope_names:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            assert len(variables) != 0
            variables += variables
        return variables


class MV3D(object):

    def __init__(self, top_shape, front_shape, rgb_shape, debug_mode=False, log_tag=None, weigths_dir=None):

        # anchors
        self.top_stride=None
        self.num_class = data.preprocess.num_class  # incude background
        if cfg.OBJ_TYPE == 'ped':
            ratios=np.array([1], dtype=np.float32)
        elif cfg.OBJ_TYPE == 'car':
            ratios = np.array([0.5, 1, 2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.float32)


        self.bases = make_bases(
            base_size = 16,
            ratios=ratios,  #aspet ratio
            scales=scales
        )

        # output dir, etc
        utilfile.makedirs(cfg.CHECKPOINT_DIR)
        self.log_msg = utilfile.Logger(cfg.LOG_DIR + '/log.txt', mode='a')
        self.track_log = utilfile.Logger(cfg.LOG_DIR + '/tracking_log.txt', mode='a')


        # creat sesssion
        self.sess = tf.Session()
        self.use_pretrain_weights=[]

        self.build_net(top_shape, front_shape, rgb_shape)

        #init subnet
        self.tag=log_tag
        self.ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, log_tag) if weigths_dir == None else weigths_dir
        self.subnet_rpn=Net(prefix='MV3D', scope_name=mv3d_net.top_view_rpn_name ,
                            checkpoint_dir=self.ckpt_dir)
        self.subnet_imfeatrue = Net(prefix='MV3D', scope_name=mv3d_net.imfeature_net_name,
                                    checkpoint_dir=self.ckpt_dir)
        self.subnet_fusion = Net(prefix='MV3D', scope_name=mv3d_net.fusion_net_name,
                                 checkpoint_dir=self.ckpt_dir)


        # set anchor boxes
        self.top_stride = self.net['top_feature_stride']
        top_feature_shape = get_top_feature_shape(top_shape, self.top_stride)
        self.top_view_anchors, self.anchors_inside_inds = make_anchors(self.bases, self.top_stride,
                                                                       top_shape[0:2], top_feature_shape[0:2])
        self.anchors_inside_inds = np.arange(0, len(self.top_view_anchors), dtype=np.int32)  # use all  #<todo>

        self.log_subdir = None
        self.top_image = None
        self.time_str = None
        self.frame_info =None


        self.batch_top_inds = None
        self.batch_top_labels =  None
        self.batch_top_pos_inds = None
        self.batch_top_targets = None
        self.batch_proposals =None
        self.batch_proposal_scores = None
        self.batch_gt_top_boxes = None
        self.batch_gt_labels = None


        # default_summary_writer
        self.default_summary_writer = None

        self.debug_mode =debug_mode

        # about tensorboard.
        self.tb_dir = log_tag if log_tag != None else strftime("%Y_%m_%d_%H_%M", localtime())

        self.log_iou_range = [range(0,2),range(2,30),range(30,60),range(60,100)]



    def predict(self, top_view, front_view, rgb_image):
        self.lables = []  # todo add lables output

        self.batch_top_view = top_view
        self.batch_rgb_images = rgb_image
        self.batch_front_view = front_view
        fd1 = {
            self.net['top_view']: self.batch_top_view,
            self.net['top_anchors']: self.top_view_anchors,
            self.net['top_inside_inds']: self.anchors_inside_inds,
            blocks.IS_TRAIN_PHASE: False,
            K.learning_phase(): True
        }

        self.batch_proposals, self.batch_proposal_scores = \
            self.sess.run([self.net['proposals'], self.net['proposal_scores']], fd1)
        self.batch_proposal_scores = np.reshape(self.batch_proposal_scores, (-1))
        self.top_rois = self.batch_proposals
        if len(self.top_rois) == 0:
            return np.zeros((0, 8, 3)), []

        self.rois3d = project_to_roi3d(self.top_rois)
        self.front_rois = project_to_front_roi(self.rois3d)
        self.rgb_rois = project_to_rgb_roi(self.rois3d)

        fd2 = {
            **fd1,
            self.net['front_view']: self.batch_front_view,
            self.net['rgb_images']: self.batch_rgb_images,

            self.net['top_rois']: self.top_rois,
            self.net['front_rois']: self.front_rois,
            self.net['rgb_rois']: self.rgb_rois,

        }

        self.fuse_probs, self.fuse_deltas = \
            self.sess.run([self.net['fuse_probs'], self.net['fuse_deltas']], fd2)

        self.probs, self.boxes3d = rcnn_nms(self.fuse_probs, self.fuse_deltas,
                                            self.rois3d,
                                            score_threshold=config.cfg.PREDICT_SCORE_THRESHOLD)

        return self.boxes3d, self.probs


    def predict_log(self, log_subdir, log_rpn=False, step=None, scope_name='',loss:tuple =None,
                    frame_tag='unknown_tag',is_train_mode=False):

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (5, 25)
        self.log_fusion_net_detail(log_subdir, self.fuse_probs, self.fuse_deltas)

        # origin top view
        self.top_image = data.draw_top_image(self.batch_top_view[0])
        top_view_log = self.top_image.copy()

        # add text on origin
        text = frame_tag
        cv2.putText(top_view_log, text, text_pos, font, 0.5, (0, 255, 100), 0, cv2.LINE_AA)
        if log_rpn:
            rpn_img = self.log_rpn(step=step, scope_name=scope_name, tensor_board=False,draw_rpn_target=False)
            top_view_log = np.concatenate((top_view_log, rpn_img), 1)

        # all prediction on top
        probs, boxes3d = rcnn_nms(self.fuse_probs, self.fuse_deltas,self.rois3d, score_threshold=0.)
        fusion_proposal_top = data.draw_box3d_on_top(self.top_image, boxes3d,scores=probs,thickness=0)
        prediction_top = data.draw_box3d_on_top(self.top_image, self.boxes3d, scores=self.probs,
                                                     thickness=0)
        # add fusion loss text
        text = ''
        if loss != None: text += 'loss c: %6f r: %6f' % loss
        cv2.putText(fusion_proposal_top, text, text_pos, font, 0.5, (0, 255, 100), 0, cv2.LINE_AA)

        # concatenate top_view_log and final prediction
        top_view_log = np.concatenate((top_view_log, fusion_proposal_top), 1)
        top_view_log = np.concatenate((top_view_log, prediction_top), 1)

        self.summary_image(top_view_log, scope_name + '/top_view', step=step)

        # prediction on rgb
        text_lables = ['No.%d class:1 prob: %.4f' % (i, prob) for i, prob in enumerate(self.probs)]
        prediction_on_rgb = nud.draw_box3d_on_camera(self.batch_rgb_images[0], self.boxes3d,
                                                          text_lables=text_lables)
        self.summary_image(prediction_on_rgb, scope_name + '/prediction_on_rgb', step=step)



    def batch_data_is_invalid(self,train_gt_boxes3d):
        # todo : support batch size >1

        for i in range(len(train_gt_boxes3d)):
            if box.box3d_in_top_view(train_gt_boxes3d[i]):
                continue
            else:
                return True
        return False


    def build_net(self, top_shape, front_shape, rgb_shape):
        with tf.variable_scope('MV3D'):
            net = mv3d_net.load(top_shape, front_shape, rgb_shape, self.num_class, len(self.bases))
            self.net = net


    def variables_initializer(self):

        # todo : remove it
        self.sess.run(tf.global_variables_initializer(),
                 {blocks.IS_TRAIN_PHASE: True, K.learning_phase(): 1})


    def load_weights(self, weights=[]):
        for name in weights:
            if name == mv3d_net.top_view_rpn_name:
                self.subnet_rpn.load_weights(self.sess)

            elif name == mv3d_net.fusion_net_name:
                self.subnet_fusion.load_weights(self.sess)

            elif name == mv3d_net.imfeature_net_name:
                self.subnet_imfeatrue.load_weights(self.sess)

            else:
                ValueError('unknow weigths name')

    def clean_weights(self, weights=[]):
        for name in weights:
            if name == mv3d_net.top_view_rpn_name:
                self.subnet_rpn.clean_weights()

            elif name == mv3d_net.fusion_net_name:
                self.subnet_fusion.clean_weights()

            elif name == mv3d_net.imfeature_net_name:
                self.subnet_imfeatrue.clean_weights()

            else:
                ValueError('unknow weigths name')


    def save_weights(self, weights=None, dir=None):
        dir = self.ckpt_dir if dir ==None else dir
        if weights==None:
            weights = [mv3d_net.top_view_rpn_name,mv3d_net.fusion_net_name,mv3d_net.imfeature_net_name]
        for name in weights:
            if name == mv3d_net.top_view_rpn_name:
                self.subnet_rpn.save_weights(self.sess,dir=os.path.join(dir, name))

            elif name == mv3d_net.fusion_net_name:
                self.subnet_fusion.save_weights(self.sess, dir=os.path.join(dir, name))

            elif name == mv3d_net.imfeature_net_name:
                self.subnet_imfeatrue.save_weights(self.sess, dir=os.path.join(dir, name))

            else:
                ValueError('unknow weigths name')


    def top_image_padding(self, top_image):
        return np.concatenate((top_image, np.zeros_like(top_image)*255,np.zeros_like(top_image)*255), 1)


    def log_rpn(self, step=None, scope_name='', loss=None, tensor_board=True, draw_rpn_target=False):

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
        
        total_img = None
        if gt_top_boxes is not None:
            total_img = draw_rpn_gt(top_image, gt_top_boxes, gt_labels)
        # nud.imsave('img_rpn_gt', img_gt, subdir)

        if draw_rpn_target:
            img_label = draw_rpn_labels(top_image, self.top_view_anchors, top_inds, top_labels)
            # nud.imsave('img_rpn_label', img_label, subdir)
            total_img = np.concatenate((total_img, img_label), 1) if total_img is not None else img_label
            img_target = draw_rpn_targets(top_image, self.top_view_anchors, top_pos_inds, top_targets)
            # nud.imsave('img_rpn_target', img_target, subdir)
            total_img = np.concatenate((total_img, img_target), 1)

        if proposals is not None:
            rpn_proposal = draw_rpn_proposal(top_image, proposals, proposal_scores)
            if loss != None:
                text = 'loss c: %6f r: %6f' % loss
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_pos = (0, 25)
                cv2.putText(rpn_proposal, text, text_pos, font, 0.5, (5, 255, 100), 0, cv2.LINE_AA)
            if total_img is not None:
                total_img = np.concatenate((total_img, rpn_proposal), 1)
            else:
                total_img = rpn_proposal
            # print('\nproposal_scores= {}\n'.format(proposal_scores))
            # nud.imsave('img_rpn_proposal', rpn_proposal, subdir)
            if tensor_board: self.summary_image(total_img, scope_name + '/top_view', step=step)
        return total_img



    def log_fusion_net_detail(self, subdir, fuse_probs, fuse_deltas):
        dir = os.path.join(cfg.LOG_DIR, subdir)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'fusion_net_detail.txt'), 'w') as info_file:
            info_file.write('index, fuse_probs, fuse_deltas\n')
            for i, prob in enumerate(fuse_probs):
                info_file.write('{}, {}, {}\n'.format(i, prob, fuse_deltas[i]))


    def summary_image(self, image, tag, summary_writer=None,step=None):

        #reduce image size
        new_size = (image.shape[1]//2, image.shape[0]//2)
        image=cv2.resize(image,new_size)

        if summary_writer == None:
            summary_writer=self.default_summary_writer

        im_summaries = []
        # Write the image to a string
        s = io.BytesIO()
        plt.imsave(s,image)

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag=tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        summary_writer.add_summary(summary, step)


    def summary_scalar(self, value, tag, summary_writer=None, step=None):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        if summary_writer == None:
            summary_writer=self.default_summary_writer
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        summary_writer.add_summary(summary, step)


class Predictor(MV3D):
    def __init__(self, top_shape, front_shape, rgb_shape, log_tag=None, weights_tag=None):
        weigths_dir= os.path.join(cfg.CHECKPOINT_DIR, weights_tag) if weights_tag!=None  else None
        MV3D.__init__(self, top_shape, front_shape, rgb_shape, log_tag=log_tag, weigths_dir=weigths_dir)
        self.variables_initializer()
        self.load_weights([mv3d_net.top_view_rpn_name, mv3d_net.imfeature_net_name, mv3d_net.fusion_net_name])

        tb_dir = os.path.join(cfg.LOG_DIR, 'tensorboard', self.tb_dir + '_tracking')
        if os.path.isdir(tb_dir):
            command = 'rm -rf %s' % tb_dir
            print('\nClear old summary file: %s' % command)
            os.system(command)
        self.default_summary_writer = tf.summary.FileWriter(tb_dir)
        self.n_log_scope = 0
        self.n_max_log_per_scope= 10


    def __call__(self, top_view, front_view, rgb_image):
        return self.predict(top_view, front_view, rgb_image)

    def dump_log(self,log_subdir, n_frame, frame_tag='unknown_tag'):
        n_start = n_frame - (n_frame % (self.n_max_log_per_scope))
        n_end = n_start + self.n_max_log_per_scope -1

        scope_name = 'predict_%d_%d' % (n_start, n_end)
        self.predict_log(log_subdir=log_subdir,log_rpn=True, step=n_frame,scope_name=scope_name,frame_tag=frame_tag)




class Trainer(MV3D):

    def __init__(self, train_set, validation_set, pre_trained_weights, train_targets, log_tag=None,
                 continue_train=False, fast_test_mode =False):
        top_shape, front_shape, rgb_shape = train_set.get_shape()
        MV3D.__init__(self, top_shape, front_shape, rgb_shape, log_tag=log_tag)
        self.train_set = train_set
        self.validation_set = validation_set
        self.train_target= train_targets

        self.train_summary_writer = None
        self.val_summary_writer = None
        self.tensorboard_dir = None
        self.summ = None
        self.iter_debug = 600
        self.n_global_step = 0
        self.validation_step = 40
        self.ckpt_save_step = 2000
        self.log_image_count=0
        self.n_iter=0
        self.fast_test_mode = fast_test_mode
        self.iou_store =[]


        # saver
        with self.sess.as_default():

            with tf.variable_scope('minimize_loss'):
                # solver
                # l2 = blocks.l2_regulariser(decay=0.0005)
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                # solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
                solver = tf.train.AdamOptimizer(learning_rate=0.001)

                # summary
                self.top_cls_loss = self.net['top_cls_loss']
                tf.summary.scalar('top_cls_loss', self.top_cls_loss)

                self.top_reg_loss = self.net['top_reg_loss']
                tf.summary.scalar('top_reg_loss', self.top_reg_loss)

                self.fuse_cls_loss = self.net['fuse_cls_loss']
                tf.summary.scalar('fuse_cls_loss', self.fuse_cls_loss)

                self.fuse_reg_loss = self.net['fuse_reg_loss']
                tf.summary.scalar('fuse_reg_loss', self.fuse_reg_loss)


                train_var_list =[]

                assert train_targets != []
                for target in train_targets:
                    # variables
                    if target == mv3d_net.top_view_rpn_name:
                        train_var_list += self.subnet_rpn.variables

                    elif target == mv3d_net.imfeature_net_name:
                        train_var_list += self.subnet_imfeatrue.variables

                    elif target == mv3d_net.fusion_net_name:
                        train_var_list += self.subnet_fusion.variables
                    else:
                        ValueError('unknow train_target name')

                # set loss
                if set([mv3d_net.top_view_rpn_name]) == set(train_targets):
                    targets_loss = 1. * self.top_cls_loss + 0.005 * self.top_reg_loss

                elif set([mv3d_net.imfeature_net_name]) == set(train_targets):
                    targets_loss = 1. * self.fuse_cls_loss + 1. * self.fuse_reg_loss

                elif set([mv3d_net.fusion_net_name]) == set(train_targets):
                    targets_loss = 1. * self.fuse_cls_loss + 1. * self.fuse_reg_loss

                elif set([mv3d_net.imfeature_net_name, mv3d_net.fusion_net_name]) == set(train_targets):
                    targets_loss = 1. * self.fuse_cls_loss + 1. * self.fuse_reg_loss

                elif set([mv3d_net.top_view_rpn_name, mv3d_net.imfeature_net_name,mv3d_net.fusion_net_name])\
                        == set(train_targets):
                    targets_loss = 1. * (1. * self.top_cls_loss + 0.05 * self.top_reg_loss) + \
                                 1. * self.fuse_cls_loss + 1. * self.fuse_reg_loss
                else:
                    ValueError('unknow train_target set')

                tf.summary.scalar('targets_loss', targets_loss)
                self.solver_step = solver.minimize(loss = targets_loss,var_list=train_var_list)


            # summary.FileWriter
            train_writer_dir = os.path.join(cfg.LOG_DIR, 'tensorboard', self.tb_dir + '_train')
            val_writer_dir = os.path.join(cfg.LOG_DIR, 'tensorboard',self.tb_dir + '_val')
            if continue_train == False:
               if os.path.isdir(train_writer_dir):
                   command ='rm -rf %s' % train_writer_dir
                   print('\nClear old summary file: %s' % command)
                   os.system(command)
               if os.path.isdir(val_writer_dir):
                   command = 'rm -rf %s' % val_writer_dir
                   print('\nClear old summary file: %s' % command)
                   os.system(command)

            graph = None if continue_train else tf.get_default_graph()
            self.train_summary_writer = tf.summary.FileWriter(train_writer_dir,graph=graph)
            self.val_summary_writer = tf.summary.FileWriter(val_writer_dir)

            summ = tf.summary.merge_all()
            self.summ = summ

            self.variables_initializer()

            #remove old weigths
            if continue_train == False:
                self.clean_weights(train_targets)

            self.load_weights(pre_trained_weights)
            if continue_train: self.load_progress()


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





    def log_fusion_net_target(self,rgb, scope_name=''):
        subdir = self.log_subdir
        top_image = self.top_image

        img_rgb_rois = box.draw_boxes(rgb, self.batch_rgb_rois[np.where(self.batch_fuse_labels == 0), 1:5][0],
                                      color=(0, 0, 255), thickness=1)
        img_rgb_rois = box.draw_boxes(img_rgb_rois,
                                      self.batch_rgb_rois[np.where(self.batch_fuse_labels != 0), 1:5][0],
                                      color=(255, 255, 255), thickness=3)
        # nud.imsave('img_rgb_rois', img_rgb_rois, subdir)
        self.summary_image(img_rgb_rois, scope_name+'/img_rgb_rois', step=self.n_global_step)

        # labels, deltas, rois3d, top_img, cam_img, class_color
        top_img, cam_img = draw_fusion_target(self.batch_fuse_labels, self.batch_fuse_targets, self.batch_rois3d,
                                              top_image, rgb, [[10, 20, 10], [255, 0, 0]])
        top_new_size = (cam_img.shape[0], int(top_img.shape[1]*(cam_img.shape[0]/top_img.shape[0])))
        top_img = cv2.resize(top_img,(top_new_size[1],top_new_size[0]))

        img_cat = np.concatenate((top_img,cam_img),1)
        self.summary_image(img_cat, scope_name+'/fusion_target', step=self.n_global_step)


    def log_prediction(self, batch_top_view, batch_front_view, batch_rgb_images,
                       batch_gt_labels=None, batch_gt_boxes3d=None, print_iou=False,
                       log_rpn=False, step=None, scope_name='',loss:tuple =None):

        boxes3d, lables = self.predict(batch_top_view, batch_front_view, batch_rgb_images)
        self.predict_log(self.log_subdir,log_rpn=log_rpn, step=step, scope_name=scope_name, loss=loss,
                         frame_tag=self.frame_id, is_train_mode=True)


        if type(batch_gt_boxes3d)==np.ndarray and type(batch_gt_labels)==np.ndarray:
            inds = np.where(batch_gt_labels[0]!=0)
            try:
                iou = box.boxes3d_score_iou(batch_gt_boxes3d[0][inds], boxes3d)
                tag = os.path.join('IOU')
                self.summary_scalar(value=iou, tag=tag, step=self.n_global_step)
            except ValueError:
                iou= -1
                print("waring :", sys.exc_info()[0])
            if print_iou: self.log_msg.write('\n %s iou: %.5f\n' % (self.step_name, iou))


    def log_info(self, subdir, info):
        dir = os.path.join(cfg.LOG_DIR, subdir)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'info.txt'), 'w') as info_file:
            info_file.write(info)

    def save_progress(self):
        print('Save progress !')
        path = os.path.join(cfg.LOG_DIR, 'train_progress',self.tag,'progress.data')
        os.makedirs(os.path.dirname(path) ,exist_ok=True)
        pickle.dump(self.n_global_step, open(path, "wb"))


    def load_progress(self):
        path = os.path.join(cfg.LOG_DIR, 'train_progress', self.tag, 'progress.data')
        if os.path.isfile(path):
            print('\nLoad progress !')
            self.n_global_step = pickle.load(open(path, 'rb'))
        else:
            print('\nCan not found progress file')



    def fit_iteration_gen(self, max_iter):
        while True:
            if 1 and iter == 0:
                self.fit_iteration( summary_it=True,summary_runmeta=True)
                yield


            elif (self.n_iter + 1) % self.iter_debug == 0 or self.fast_test_mode:
                ## log iou and image
                for i in range(len(self.log_iou_range) * 3):
                    self.fit_iteration(is_validation=False, log_image=True, iou_statistic=True)
                    yield
                self.fit_iteration(is_validation=False, summary_it=True, log_image=True, summary_iou=True)

                # for validation
                for i in range(len(self.log_iou_range) * 3):
                    self.fit_iteration(is_validation=True, log_image=True, iou_statistic=True)
                    yield
                self.fit_iteration(is_validation=True, summary_it=True, log_image=True, summary_iou=True)
                yield
                print('log iou and image done')


            # summary loss
            elif self.n_iter % self.validation_step == 0:
                # train
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                    self.fit_iteration(is_validation=False, summary_it=True)

                self.log_msg.write('%10s: |  %5d/%5d  %0.5f   %0.5f   |   %0.5f   %0.5f \n' % \
                                   ('training', self.n_iter, max_iter, t_cls_loss, t_reg_loss,
                                    f_cls_loss, f_reg_loss))

                # validation
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                    self.fit_iteration(is_validation=True, summary_it=True)
                self.log_msg.write('%10s: |  %5d/%5d  %0.5f   %0.5f   |   %0.5f   %0.5f \n' % \
                                   ('validation', self.n_iter, max_iter, t_cls_loss, t_reg_loss,
                                    f_cls_loss, f_reg_loss))
                yield

            else:
                # train
                self.fit_iteration()
                yield


    def __call__(self, max_iter=1000, train_set =None, validation_set =None):

        sess = self.sess
        net = self.net

        with sess.as_default():
            #for init model

            batch_size=1

            if cfg.TRAINING_TIMER:
                time_it = timer()

            # start training here  #########################################################################################
            self.log_msg.write('iter |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  total |  \n')
            self.log_msg.write('-------------------------------------------------------------------------------------\n')

            fit_iteration =self.fit_iteration_gen(max_iter)
            for iter in range(max_iter):
                self.n_iter=iter

                next(fit_iteration)

                if (iter+1)%self.ckpt_save_step==0 or self.fast_test_mode:
                    self.save_weights(self.train_target)
                    self.save_progress()

                if (iter+1)%200==0:
                    if cfg.TRAINING_TIMER:
                        self.log_msg.write('It takes %0.2f secs to train %d iterations. \n' % \
                                           (time_it.time_diff_per_n_loops(), 200))
                self.n_global_step += 1


            if cfg.TRAINING_TIMER:
                self.log_msg.write('It takes %0.2f secs to train the dataset. \n' % \
                                   (time_it.total_time()))

            self.save_weights(self.train_target)
            self.save_progress()




    def fit_iteration(self, is_validation =False, summary_it=False, summary_runmeta=False, log_image=False,
                      summary_iou=False, iou_statistic=False):

        data_set = self.validation_set if is_validation else self.train_set
        self.default_summary_writer = \
            self.val_summary_writer if is_validation else self.train_summary_writer

        self.step_name = 'validation' if is_validation else 'training'

        # load dataset
        self.batch_rgb_images, self.batch_top_view, self.batch_front_view, \
        self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id = \
            data_set.load()

        # fit_iterate log init
        if log_image:
            self.time_str = strftime("%Y_%m_%d_%H_%M", localtime())
            self.frame_info = data_set.get_frame_info()
            self.log_subdir = self.step_name + '/' + self.time_str
            self.top_image = data.draw_top_image(self.batch_top_view[0])
            # self.top_image = self.top_image_padding(top_image)


        net = self.net
        sess = self.sess

        # put tensorboard inside
        top_cls_loss = net['top_cls_loss']
        top_reg_loss = net['top_reg_loss']
        fuse_cls_loss = net['fuse_cls_loss']
        fuse_reg_loss = net['fuse_reg_loss']


        self.batch_gt_top_boxes = data.box3d_to_top_box(self.batch_gt_boxes3d[0])

        ## run propsal generation
        fd1 = {
            net['top_view']: self.batch_top_view,
            net['top_anchors']: self.top_view_anchors,
            net['top_inside_inds']: self.anchors_inside_inds,

            blocks.IS_TRAIN_PHASE: True,
            K.learning_phase(): 1
        }
        self.batch_proposals, self.batch_proposal_scores, self.batch_top_features = \
            sess.run([net['proposals'], net['proposal_scores'], net['top_features']], fd1)

        ## generate  train rois  for RPN
        self.batch_top_inds, self.batch_top_pos_inds, self.batch_top_labels, self.batch_top_targets = \
            rpn_target(self.top_view_anchors, self.anchors_inside_inds, self.batch_gt_labels[0],
                       self.batch_gt_top_boxes)

        self.batch_top_rois, self.batch_fuse_labels, self.batch_fuse_targets = \
            fusion_target(self.batch_proposals, self.batch_gt_labels[0],
                          self.batch_gt_top_boxes, self.batch_gt_boxes3d[0])

        self.batch_rois3d = project_to_roi3d(self.batch_top_rois)
        self.batch_front_rois = project_to_front_roi(self.batch_rois3d)
        self.batch_rgb_rois = project_to_rgb_roi(self.batch_rois3d)


        ## run classification and regression loss -----------
        fd2 = {
            **fd1,

            net['top_view']: self.batch_top_view,
            net['front_view']: self.batch_front_view,
            net['rgb_images']: self.batch_rgb_images,

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

        if self.debug_mode:
            print('\n\nstart debug mode\n\n')
            debug_sess=tf_debug.LocalCLIDebugWrapperSession(sess)
            t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                debug_sess.run([top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss], fd2)


        if summary_it:
            run_options = None
            run_metadata = None

            if is_validation:
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss, tb_sum_val = \
                    sess.run([top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss, self.summ], fd2)

                self.val_summary_writer.add_summary(tb_sum_val, self.n_global_step)
                # print('added validation  summary ')
            else:
                if summary_runmeta:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                _, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss, tb_sum_val = \
                    sess.run([self.solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss,
                              self.summ], feed_dict=fd2, options=run_options, run_metadata=run_metadata)

                self.train_summary_writer.add_summary(tb_sum_val, self.n_global_step)
                # print('added training  summary ')

                if summary_runmeta:
                    self.train_summary_writer.add_run_metadata(run_metadata, 'step%d' % self.n_global_step)
                    # print('added runtime metadata.')

        else:
            if is_validation:
                t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                    sess.run([top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss], fd2)
            else:

                _, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                    sess.run([self.solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],
                             feed_dict=fd2)

        if log_image:
            # t0 = time.time()
            step_name = 'validation' if is_validation else  'train'
            scope_name = '%s_iter_%06d' % (step_name, self.n_global_step - (self.n_global_step % self.iter_debug))

            boxes3d, lables = self.predict(self.batch_top_view, self.batch_front_view, self.batch_rgb_images)

            if 0 and set(self.train_target) !=set(mv3d_net.top_view_rpn_name):
                # get iou
                iou = -1
                inds = np.where(self.batch_gt_labels[0] != 0)
                try:
                    iou = box.boxes3d_score_iou(self.batch_gt_boxes3d[0][inds], boxes3d)
                    if iou_statistic: self.iou_store.append(iou)
                    if summary_iou:
                        iou_aver = sum(self.iou_store)/len(self.iou_store)
                        self.iou_store=[]
                        tag = os.path.join('IOU')
                        self.summary_scalar(value=iou_aver, tag=tag, step=self.n_global_step)
                        self.log_msg.write('\n %s iou average: %.5f\n' % (self.step_name, iou_aver))
                except ValueError:
                    # print("waring :", sys.exc_info()[0])
                    pass

                #set scope name
                if iou == -1:
                    scope_name = os.path.join(scope_name, 'iou_error'.format(range(5, 8)))
                else:
                    for iou_range in self.log_iou_range:
                        if int(iou*100) in iou_range:
                            scope_name = os.path.join(scope_name , 'iou_{}'.format (iou_range))

                # print('Summary log image, scope name: {}'.format(scope_name))

                self.log_fusion_net_target(self.batch_rgb_images[0], scope_name=scope_name)
            log_info_str = 'frame info: ' + self.frame_info + '\n'
            log_info_str += self.anchors_details()
            log_info_str += self.rpn_poposal_details()
            self.log_info(self.log_subdir, log_info_str)


            self.predict_log(self.log_subdir, log_rpn=True, step=self.n_global_step,
                             scope_name=scope_name, loss=(f_cls_loss, f_reg_loss),
                             frame_tag=self.frame_id, is_train_mode=True)


            # self.log_msg.write('Image log  summary use time : {}\n'.format(time.time() - t0))


        return t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss




