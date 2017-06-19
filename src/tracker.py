import os
os.environ['DISPLAY'] = ':0'
import tensorflow as tf
tf.set_random_seed(7)
import numpy as np
np.random.seed(7)
from net.blocks import *
from tensorflow.contrib import rnn
from time import strftime,localtime
from config import cfg
import utils.batch_loading as ub
import net.processing.boxes3d as boxes3d

n_max_objects_output = 1
n_max_objects_input = 5
ground_truth =np.c_[np.array(range(0,100)),np.ones((100)), np.ones((100))*2].astype(np.float32)


def get_noise(n_max,probality=0.4):
    n_noise= 0
    if np.random.random()>1- probality:
        n_noise = int(np.random.random() * n_max)

    noise_list = [np.random.random((3))*100 for  i in range(n_noise)]
    return np.array(noise_list)

def get_train_data(idx,n_max_objects):
    obj_miss_probality = 0.5
    data_trans_gt=[]

    if np.random.random()> obj_miss_probality:
        data_trans_gt = ground_truth[j:j + 1, :] +np.random.random(3)*2
    noise = get_noise(n_max_objects_input-1)

    n_objs = len(noise)+len(data_trans_gt)
    padding = np.zeros((n_max_objects-n_objs,3))

    data_trans_noised=padding

    #
    if len(data_trans_gt)!=0:
        data_trans_noised=np.r_[data_trans_gt,padding]

    # add noise
    if len(noise)!=0:
        data_trans_noised = np.r_[data_trans_noised,noise]

    np.random.shuffle(data_trans_noised)
    data_trans_noised = data_trans_noised.reshape((1, 1, n_max_objects, 3))
    return data_trans_noised.astype(np.float32), n_objs

def get_train_data2(n_max_objects,data_trans_gt):
    obj_miss_probality = 0.2

    if np.random.random()> obj_miss_probality:
        data_trans_noised = data_trans_gt + np.random.random(3)*2
    else:
        data_trans_noised = np.random.random((1,3)) * 2
    noise = get_noise(n_max_objects_input-1)

    n_objs = len(noise)+len(data_trans_noised)
    padding = np.zeros((n_max_objects-n_objs,3))

    if len(padding)!=0:
        data_trans_noised=np.r_[data_trans_noised, padding]


    # add noise
    if len(noise)!=0:
        data_trans_noised = np.r_[data_trans_noised,noise]

    np.random.shuffle(data_trans_noised)
    data_trans_noised = data_trans_noised.reshape((1, 1, n_max_objects, 3))
    return data_trans_noised, n_objs

class Validation(object):



    def __init__(self, sess, net_trans_history ,net_trans_gt, net_rnn_states_c, net_rnn_states_h):
        self.sess = sess
        time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
        self.summary_writer = tf.summary.FileWriter(os.path.join(cfg.LOG_DIR, 'tensorboard/%s_tracker_val' %
                                                            (time_str)), sess.graph)

        validation_dataset = {
            '1': ['21_f'],
            '3': ['7', '11_f']
        }

        self.val_set_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, validation_dataset)

        self.last_states_c = np.zeros((1, n_hidden))
        self.last_states_h = np.zeros((1, n_hidden))

        self.net_trans_history = net_trans_history,
        self.net_trans_gt =net_trans_gt,
        self.net_rnn_states_c =net_rnn_states_c,
        self.net_rnn_states_h= net_rnn_states_h,

    def run(self, iter):
        # load dataset from didi
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, \
        frame_id = self.val_set_loader.load(1)

        # '1/15/00000'.split('/')[2] == '00000':
        if frame_id[0].split('/')[2] == '00000':
            self.last_states_c = np.zeros((1, n_hidden))
            self.last_states_h = np.zeros((1, n_hidden))
            n_history_saved = 0

        trans_gt, size, rotation = boxes3d.boxes3d_decompose(train_gt_boxes3d[0])
        trans_history, _ = get_train_data2(n_max_objects_input, trans_gt)

        feed = {
            self.net_trans_history: trans_history,
            self.net_trans_gt: trans_gt,
            self.net_rnn_states_c: self.last_states_c,
            self.net_rnn_states_h: self.last_states_h,
            IS_TRAIN_PHASE: True
        }

        self.last_states_c, self.last_states_h = last_states.c, last_states.h

        summary, total_loss, trans_reg_loss, error_reg_loss, predict_tans, error_predict, = \
            sess.run([summary_op, net_loss, net_trans_reg_loss, net_score_reg_loss,
                      net_trans_predict, net_score_predict], feed)
        self.summary_writer.add_summary(summary, iter)

        # print('val loss: {} {} {}  | predict: {} gt: {} score:{} frame: {}\n{}\n\n'.
        #       format(total_loss, trans_reg_loss, error_reg_loss, predict_tans, trans_gt,
        #              error_predict, frame_id, trans_history))



if __name__ == '__main__':

    model_num = 1
    n_steps = 1  # timesteps
    n_hidden = 64
    error_threshold = 10

    #top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
    net_trans_history = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_max_objects_input, 3], name='trans_history')
    net_trans_gt = tf.placeholder(dtype=tf.float32, shape=[None, 3],name= 'trans_history')
    net_rnn_states_c = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden], name='rnn_states_c')
    net_rnn_states_h = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden], name='rnn_states_h')
    net_last_states = tf.contrib.rnn.LSTMStateTuple(net_rnn_states_c, net_rnn_states_h)




    # [batch_size, sequence_length_max, vector_size]
    input = tf.reshape(net_trans_history, [-1, 1, n_max_objects_input*3])

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = rnn.DropoutWrapper(lstm_cell,input_keep_prob=0.5, output_keep_prob=0.5)

    # Get lstm cell output
    outputs, net_states = tf.nn.dynamic_rnn(lstm_cell, input,initial_state=net_last_states, dtype=tf.float32)

    block = outputs[:,-1,:]
    # [translation,confidence]
    net_predict = linear(block, num_hiddens=(3+1) * n_max_objects_output, name='fc_3')
    net_trans_predict = net_predict[:,0:3]
    net_score_predict = tf.abs(net_predict[:, 3])
    sign = tf.cast(tf.less_equal(net_score_predict , 1), tf.float32)
    net_score_predict = sign * net_score_predict

    # obj_trans = tf.reshape(obj_trans, tf.nn.dynamic_rnn(lstm_cell, net_trans_history,initial_state=last_states, dtype=tf.float32)[None,n_max_objects,3],'reshape liner output')
    detal = net_trans_predict - net_trans_gt
    detal_l2 = tf.sqrt(tf.reduce_mean(tf.reduce_sum(detal * detal, axis=1)))
    net_trans_reg_loss = detal_l2

    # score
    sign = tf.cast(tf.less(detal_l2, error_threshold), tf.float32)
    net_score_gt = sign * (1 - detal_l2 / error_threshold)
    net_score_reg_loss = tf.reduce_mean(tf.abs(net_score_predict - net_score_gt))

    net_loss = net_trans_reg_loss
    # net_loss = net_trans_reg_loss

    # solver
    solver = tf.train.AdamOptimizer(learning_rate=0.001)
    solver_step = solver.minimize(net_loss)

    sess = tf.Session()
    time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    summary_writer = tf.summary.FileWriter(os.path.join(cfg.LOG_DIR, 'tensorboard/%s_tracker_tra' %
                                                        (time_str)), sess.graph)

    tf.summary.scalar('total_loss', net_loss)
    tf.summary.scalar('net_trans_reg_loss', net_trans_reg_loss)
    tf.summary.scalar('net_score_reg_loss', net_score_reg_loss)
    summary_op = tf.summary.merge_all()

    # dataset loader

    training_dataset = {
        '1': ['6_f', '9_f', '15', '20'],
        '2': ['3_f'],
        '3': ['2_f', '4', '6', '8', '7']}


    tra_set_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, training_dataset)
    saver = tf.train.Saver()

    validation= Validation(sess, net_trans_history, net_trans_gt, net_rnn_states_c, net_rnn_states_h)
    with sess.as_default():
        os.makedirs(os.path.join(cfg.CHECKPOINT_DIR,'tracker_net'),exist_ok=True)
        pretrained_model_path = os.path.join(cfg.CHECKPOINT_DIR,'tracker_net', 'tracker.ckpt')
        if 0 and tf.train.checkpoint_exists(pretrained_model_path):
            print('load pretrained model')
            saver.restore(sess, pretrained_model_path)
        else:
            sess.run( tf.global_variables_initializer(),feed_dict={IS_TRAIN_PHASE: True})

        last_states_c = np.zeros((1, n_hidden))
        last_states_h = np.zeros((1, n_hidden))

        for i in range(20000):

            # load dataset from didi
            train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d,\
            frame_id = tra_set_loader.load(1)

            #'1/15/00000'.split('/')[2] == '00000':
            if frame_id[0].split('/')[2] == '00000':
                last_states_c = np.zeros((1, n_hidden))
                last_states_h = np.zeros((1, n_hidden))
                n_history_saved = 0

            trans_gt, size, rotation= boxes3d.boxes3d_decompose(train_gt_boxes3d[0])
            trans_history, _ = get_train_data2(n_max_objects_input,trans_gt )


            feed={
                net_trans_history: trans_history,
                net_trans_gt: trans_gt,
                net_rnn_states_c: last_states_c,
                net_rnn_states_h: last_states_h,
                IS_TRAIN_PHASE: True
            }


            _, total_loss,trans_reg_loss,error_reg_loss,predict_tans,error_predict,last_states = \
                sess.run([solver_step,net_loss, net_trans_reg_loss, net_score_reg_loss,
                          net_trans_predict, net_score_predict, net_states], feed)

            last_states_c,last_states_h=last_states.c,last_states.h
            if np.random.random() < (1./20):
                summary,total_loss, trans_reg_loss, error_reg_loss, predict_tans, error_predict,  = \
                    sess.run([summary_op, net_loss, net_trans_reg_loss, net_score_reg_loss,
                              net_trans_predict, net_score_predict], feed)
                summary_writer.add_summary(summary,i)
                validation.run(i)

            if i%200==0:
                print('loss: {} {} {}  | predict: {} gt: {} score:{} frame: {}\n{}\n\n'.
                      format( total_loss,trans_reg_loss,error_reg_loss, predict_tans, trans_gt,
                                                          error_predict,frame_id,trans_history))
            if i>=800 and i % 800 == 0:
                saver.save(sess, pretrained_model_path)
                print('saved model')




