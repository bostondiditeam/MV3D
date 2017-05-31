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
    obj_miss_probality = 0.3
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


if __name__ == '__main__':

    model_num = 1
    n_steps = 1  # timesteps
    n_hidden = 256

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

    # Get lstm cell output
    outputs, net_states = tf.nn.dynamic_rnn(lstm_cell, input,initial_state=net_last_states, dtype=tf.float32)

    block = outputs[:,-1,:]
    net_trans_predic = linear(block, num_hiddens=3 * n_max_objects_output, name='fc_1')
    # obj_trans = tf.reshape(obj_trans, tf.nn.dynamic_rnn(lstm_cell, net_trans_history,initial_state=last_states, dtype=tf.float32)[None,n_max_objects,3],'reshape liner output')
    detal = net_trans_predic - net_trans_gt
    net_loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(detal * detal, axis=1)))

    solver = tf.train.AdamOptimizer(learning_rate=0.0001)
    solver_step = solver.minimize(net_loss)

    sess = tf.Session()
    # time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    # summary_writer = tf.summary.FileWriter(os.path.join(cfg.LOG_DIR, 'tensorboard/%s_model_%d' %
    #                                                     (time_str, model_num)), sess.graph)
    #
    # tf.summary.scalar('loss', net_loss)
    # summary_op = tf.summary.merge_all()

    with sess.as_default():
        sess.run( tf.global_variables_initializer())
        for i in range(20000):
            last_states_c= np.zeros((1,n_hidden))
            last_states_h = np.zeros((1, n_hidden))


            for j in range(len(ground_truth)):
                trans_gt = np.reshape(ground_truth[j,:], (1, 3))
                trans_history,_ = get_train_data(j,n_max_objects_input)
                feed={
                    net_trans_history: trans_history,
                    net_trans_gt: trans_gt,
                    net_rnn_states_c: last_states_c,
                    net_rnn_states_h: last_states_h
                }

                _, loss,predict_tans,last_states = sess.run([solver_step, net_loss , net_trans_predic,
                                                                net_states],feed)
                last_states_c,last_states_h=last_states.c,last_states.h
                if i%10==1 and j==int(np.random.random()*100):
                    print('{} {} {} \n{}\n\n'.format( loss, predict_tans, trans_gt,trans_history))
                    # summary_writer.add_summary(summary,i)



