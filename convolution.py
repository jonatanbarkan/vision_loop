from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import style
from functions import downgrade
from functions import rotate_image_180


cap = cv2.VideoCapture('S06E02.mkv')
# cap.set(1, 310)
plt.ion()
style.use('ggplot')
fig = plt.figure()
# initialize weights
shape = (15, 15, 3)
weights = np.random.normal(0.0, 0.5, shape)
# weights = np.ones(shape)/(shape[0] * shape[1])

# weights = np.zeros(shape)
# weights[7, 7, 0] = 0.5
# weights[7, 7, 1] = 0.5
# weights[7, 7, 2] = 0.5
# weights[1, 3, 0] = 1
# weights[9, 3, 1] = 1

alpha = 0.05

# Capture frame-by-frame
ret, frame = cap.read()
I_t = downgrade(frame, 320)
I_t = I_t.astype(float)/255
frame_shape = I_t.shape[0] * I_t.shape[1]

# initialize captures
time = 0
times = []
errors = []

# start looping on frames from scene
while ret:  # time step, ret = False means frame I_t is None

    ret, frame = cap.read()
    if frame is None:
        break
    I_t1 = downgrade(frame, 320)
    I_t1 = I_t1.astype(float) / 255
    # ------------------------------------------------------------------------------------------
    # convolve I_t and weights:
    # prediction = conv(I_t, weights)
    prediction = cv2.filter2D(I_t, -1, weights)
    epsilon = prediction - I_t  # not padded with zeroes
    padd_and_rotate_image = cv2.copyMakeBorder(rotate_image_180(I_t), shape[0], 0, shape[1], 0, cv2.BORDER_CONSTANT,
                                               value=(0, 0, 0, 0))
    # # temp = conv(weights, epsilon) with padded zeroes
    # temp = cv2.filter2D(weights, -1, padded_epsilon, borderType=cv2.BORDER_CONSTANT) / (frame_shape)
    temp = cv2.filter2D(epsilon, -1, padd_and_rotate_image, borderType=cv2.BORDER_CONSTANT) / (frame_shape)
    dw = temp[0:shape[0], 0:shape[1]]
    # weights += alpha*temp
    weights -= alpha * dw

    # error calculation

    error = np.mean(np.power(np.ravel(epsilon), 2))

    times.append(time)
    errors.append(np.log(error))

    # ------------------------------------------------------------------------------------------
    # ret = False
    # cv2.imshow('frame', I_t)
    # if time % 10 == 9:
    if time % 10 == 9:
        plt.gcf().clear()
        fig.suptitle('conv{0}'.format(time))
        plt.subplot(321), plt.imshow(weights), plt.title('weights')
        plt.xticks([]), plt.yticks([])
        plt.subplot(322), plt.imshow(I_t), plt.title('I_t')
        plt.xticks([]), plt.yticks([])
        plt.subplot(323), plt.imshow(prediction), plt.title('prediction')
        plt.xticks([]), plt.yticks([])
        plt.subplot(324), plt.imshow(temp), plt.title('temp')
        plt.xticks([]), plt.yticks([])
        plt.subplot(325), plt.imshow(epsilon), plt.title('epsilon')
        plt.xticks([]), plt.yticks([])
        plt.subplot(326), plt.imshow(dw), plt.title('dw')
        plt.xticks([]), plt.yticks([])
        plt.pause(0.00001)
        plt.show()
    # if time % 1000 == 999:
    #     plt.plot(times, errors)
    #     plt.xlabel('time step')
    #     plt.ylabel('error rate')
    #     plt.pause(0.0001)
    #     plt.show()
    #     print('.',)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    I_t = I_t1
    time += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


plt.plot(times, errors)
plt.xlabel('time step')
plt.ylabel('log error rate')
plt.title('constant weights')
plt.show()
plt.savefig('predict_current_constant_weights_2')
