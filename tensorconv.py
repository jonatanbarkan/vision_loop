from __future__ import division
from functions import *
import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib import style


style.use('ggplot')

train_dir = 'same'

cap = cv2.VideoCapture('S06E02.mkv')
# cap.set(1, 310)
_, frame = cap.read()
# last_frame = downgrade(last_frame, 320)

# ---------------------------------------- hyperparameters ---------------------------------------------------------
frame_jump = 3
fps = int(cap.get(5)/frame_jump)
wanted_width = 320  # the width of the frame we want. the height is calculated from it to keep scale
wanted_height = int(frame.shape[0] * wanted_width / frame.shape[1])
num_output = 1  # if we want a number diff then 1 and three then need to change epsilon calculation
learning_rate = 0.001  # learning rate of SGD
patch_width = 35  # the width of the sliding window frame
patch_height = 35  # the height of the sliding window frame

# ---------------------------------------- implementation ---------------------------------------------------------

patch_size = [patch_width, patch_height]
patch_channels = [3]  # num of channels - here it's blue green and red ie bgr
patch_num_features = [num_output]  # num of output features - here we want only 1 to get a kernel, if 3 then outputs kernel to every channel

loop_name = 'train_blur{0}x{1}_lr{2}_fps{3}'.format(patch_width, patch_height, learning_rate, fps)

# ---------------------------------------- graph building ---------------------------------------------------------


with tf.name_scope('input'):
    input_image = tf.placeholder(dtype=tf.float32, name='flat')
    batch_input = tf.reshape(input_image, [-1, wanted_height, wanted_width, 3])
    # input_image = tf.placeholder(dtype=tf.float32, name='original', shape=(181, 320, 3))
    # batch_input = tf.expand_dims(input_image, dim=0)
# input_resized = tf.image.resize_area(input_image, size=[0, 180, 320, 3], name='resize')  # resizing with opencv
# with tf.name_scope('y_true'):
#     output_image = tf.placeholder(dtype=tf.float32, name='flat')
#     batch_output = tf.reshape(output_image, [-1, wanted_height, wanted_width, 3])
    # output_image = tf.placeholder(dtype=tf.float32, name='original')
    # batch_output = tf.expand_dims(output_image, dim=0)

with tf.name_scope('conv_layer'):

    with tf.name_scope('weights'):
        weights = weight_variable(shape=patch_size+patch_channels+patch_num_features)
        variable_summaries(weights, name='weights')

    with tf.name_scope('bias'):
        bias = bias_variable(shape=patch_num_features)
        variable_summaries(bias, name='bias')

    with tf.name_scope('y_pred'):
        prediction = forward_conv2d(batch_input, weights, bias, name='prediction')
        variable_summaries(prediction, name='prediction')

with tf.name_scope('error'):
    # with tf.name_scope(self.name + '_' + 'total'):
    with tf.name_scope('epsilon'):
        epsilon = tf.sub(prediction, batch_input, name='epsilon')
    cost = tf.reduce_mean(tf.square(epsilon))
    tf.scalar_summary('cost', cost)

# global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# ---------------------------------------- run program ---------------------------------------------------------

plt.ion()
init = tf.initialize_all_variables()
# graph = tf.Graph().as_default()
with tf.Session() as sess:
    # ---------------------------------------- init session -----------------------------------------------------
    summary_op = tf.merge_all_summaries()
    filter_vis = tf.image_summary('filter', visualize(weights), max_images=20)
    summary_writer = tf.train.SummaryWriter(train_dir + '/' + loop_name, sess.graph)
    sess.run(init)

    # plt.ion()
    cap = cv2.VideoCapture('S06E02.mkv')
    # cap.set(1, 310)
    ret, frame = cap.read()
    # last_frame = downgrade(last_frame, 320)
    time_step = 0

    while ret:  # time step

        # last_frame = frame
        frame = downgrade(frame, wanted_width)
        frame = frame.astype(float)/255
        flat_input = np.ravel(frame)
        # ------------------------------------------------------------------------------------------
        feed_dict = {input_image: flat_input}
        if time_step % 1000 == 999:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, summary, cost_value = sess.run([train_step, summary_op, cost], feed_dict=feed_dict, options=run_options,
                                              run_metadata=run_metadata)
            summary_writer.flush()
            summary_writer.add_run_metadata(run_metadata, 'step%03d' % time_step)
        else:
            _, summary, vis, cost_value = sess.run([train_step, summary_op, filter_vis, cost], feed_dict=feed_dict)
            # if time_step % 100 == 0:
            #     W = np.squeeze(sess.run(weights))
            #     plt.title('weights {0}'.format(int(time_step / 100)))
            #     plt.imshow(W)
            #     plt.grid(False)
            #     plt.pause(0.00001)
            #     plt.show()
            #     print(np.mean(W),)

            if time_step % 150 == 0:
                # W = weights.eval(sess)
                # print np.mean(W), np.amax(W), np.amin(W)
                # plt.title('weights in step/n{0}'.format(time_step))
                # plt.imshow(np.squeeze(W))
                # plt.xticks([]), plt.yticks([])
                # plt.pause(0.00001)
                # plt.show()
                summary_writer.flush()
                summary_writer.add_summary(vis, time_step)
                summary_writer.add_summary(summary, time_step)
                # print('cost = {0}, step = {1} , '.format(cost_value, time_step),)
        # ------------------------------------------------------------------------------------------
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_step += 1
        cap.set(1, cap.get(1) + frame_jump)
        ret, frame = cap.read()

    summary_writer.close()
    # sess.close()
    cap.release()
    cv2.destroyAllWindows()


