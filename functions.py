from __future__ import division
import cv2
import numpy as np
import tensorflow as tf


def downgrade(image, target_res_width):
    target_res_width = int(target_res_width)
    r = target_res_width / image.shape[1]
    dim = (target_res_width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def rotate_image_180(image):
    # grab the dimensions of the image and calculate the center
    # of the image
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def vis_conv(v,ix,iy,ch,cy,cx, p = 0) :
    v = np.reshape(v,(iy,ix,ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy,ix,cy,cx))
    v = np.transpose(v,(2,0,3,1)) #cy,iy,cx,ix
    v = np.reshape(v,(cy*iy,cx*ix))
    return v


def regularize(frame):
    frame = np.divide(frame, [255])
    return frame


def visualize(W):
    # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
    x_min = tf.reduce_min(W)
    x_max = tf.reduce_max(W)
    weights_0_to_1 = tf.div(tf.sub(W, x_min), tf.sub(x_max, x_min))
    weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)

    # to tf.image_summary format [batch_size, height, width, channels]
    weights_transposed = tf.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])
    return weights_transposed


def weight_variable(shape):
    # initial = tf.zeros(shape, dtype=tf.float32)
    # initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    # initial = tf.zeros(shape, dtype=tf.float32)
    initial = tf.div(tf.ones(shape, dtype=tf.float32), shape[0]*shape[1])
    return tf.Variable(initial, trainable=True)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def forward_conv2d(x, W, b, stride_size=1, act = tf.nn.relu,  name=''):
    with tf.name_scope('convolution'):
        conv = tf.nn.conv2d(x, W, strides=[1, stride_size, stride_size, 1], padding='SAME', name=name+'convolution')
        variable_summaries(conv, 'convolution')
    with tf.name_scope('preactivate'):
        preactivate = conv + b
        variable_summaries(preactivate, 'preactivate')
    with tf.name_scope('activate'):
        activate = tf.minimum(tf.maximum(conv, 0),1, name=name+'activation')
    return activate


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope(name + '_' + 'summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary(name + '_' + 'mean/' + name, mean)
        with tf.name_scope(name + '_' + 'stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(var, mean))))
        tf.scalar_summary(name + '_' + 'sttdev/' + name, stddev)
        tf.scalar_summary(name + '_' + 'max/' + name, tf.reduce_max(var))
        tf.scalar_summary(name + '_' + 'min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)