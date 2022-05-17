# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os, glob
import time

from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Add, MaxPooling2D, Activation, Dense, Reshape, \
    GlobalAveragePooling2D, Multiply, Conv2DTranspose, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from preprocessing import utils
import tensorflow as tf
from utils import *

tf.random.set_seed('seed')

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

"""
# identify GPU
device_name = tf.test.gpu_device_name()
if not tf.config.list_physical_devices('GPU'):
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.75)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
tf.compat.v1.keras.backend.set_session(sess)
"""

img_rows = 256
img_cols = 256


def squeeze_excite_block(inputs, ratio = 8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False)(se)
    se = Dense(filters, activation = 'sigmoid', kernel_initializer = 'he_normal', use_bias = False)(se)

    x = Multiply()([init, se])
    return x


def resnet_block(x, n_filter, strides = 1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filter, (3, 3), padding = "same", strides = strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding = "same", strides = 1)(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding = "same", strides = strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def get_rwnet():
    inputs = Input(shape = (img_rows, img_cols, 3))

    conv1 = resnet_block(inputs, 32, strides = 1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)

    conv2 = resnet_block(pool1, 64, strides = 1)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = resnet_block(pool2, 128, strides = 1)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

    conv4 = resnet_block(pool3, 256, strides = 1)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = aspp_block(pool4, 512)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(conv5), conv4], axis = 3)
    conv6 = resnet_block(up6, 256, strides = 1)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv3], axis = 3)
    conv7 = resnet_block(up7, 128, strides = 1)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv2], axis = 3)
    conv8 = resnet_block(up8, 64, strides = 1)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv8), conv1], axis = 3)
    conv9 = resnet_block(up9, 32, strides = 1)

    down10 = concatenate([Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv9), conv9], axis = 3)
    conv10 = resnet_block(down10, 32, strides = 1)
    pool10 = MaxPooling2D(pool_size = (2, 2))(conv10)

    down11 = concatenate([Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(pool10), conv8], axis = 3)
    conv11 = resnet_block(down11, 64, strides = 1)
    pool11 = MaxPooling2D(pool_size = (2, 2))(conv11)

    down12 = concatenate([Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool11), conv7], axis = 3)
    conv12 = resnet_block(down12, 128, strides = 1)
    pool12 = MaxPooling2D(pool_size = (2, 2))(conv12)

    down13 = concatenate([Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool12), conv6], axis = 3)
    conv13 = resnet_block(down13, 256, strides = 1)
    pool13 = MaxPooling2D(pool_size = (2, 2))(conv13)

    conv14 = aspp_block(pool13, 512)

    up15 = concatenate([Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(conv14), conv13], axis = 3)
    conv15 = resnet_block(up15, 256, strides = 1)

    up16 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv15), conv12], axis = 3)
    conv16 = resnet_block(up16, 128, strides = 1)

    up17 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv16), conv11], axis = 3)
    conv17 = resnet_block(up17, 64, strides = 1)

    up18 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv17), conv10], axis = 3)
    conv18 = resnet_block(up18, 32, strides = 1)

    conv18 = aspp_block(conv18, 32)

    conv19 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv18)

    model = Model(inputs = [inputs], outputs = [conv19])
    model.compile(optimizer = optimizers.adam_v2.Adam(learning_rate = 1e-4), loss = [loss],
                  metrics = [dice_coef, iou_coef])
    return model


def main():
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_rwnet()

    model.summary()

    name = 'mydata'

    # fname = 'rwnet_cbis_join_weights.h5'
    fname = '../unet/' + name + '_weights.h5'
    pred_dir = fname[:-11]

    model_checkpoint = ModelCheckpoint(fname, monitor = 'val_loss', verbose = 1, save_best_only = True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test, imgs_mask_test_gt = utils.getDatasetArraysForNet()

    history = model.fit(imgs_train, imgs_mask_train, batch_size = 8, epochs = 100, verbose = 1,
                        callbacks = [model_checkpoint],
                        validation_data = (imgs_val, imgs_mask_val),
                        shuffle = True)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.save_weights('CBIS/test')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test = model.predict(imgs_test, verbose = 1)
    np.save('imgs_mask_test_' + name + '_wunet.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    # data_path2 = 'D:/Files/MYDATA/Breast_Cancer-Begonya/Images/Test_Seg/'
    data_path2 = 'D:/INbreast/Test_Seg/'
    # data_path2 = 'D:/CBIS_augmented/Test_Seg/'
    # data_path2 = 'D:/CSAW-S/CsawS/Test_Seg/'

    d = data_path2 + 'roi/*.png'
    files = glob.glob(d)

    files1 = files

    data_path2 = 'D:/Files/MYDATA/Breast_Cancer-Begonya/Images/Test_Seg/'

    d = data_path2 + 'roi/*.png'
    files = glob.glob(d)

    files2 = files

    files = files1 + files2

    files = [file.split('\\')[-1][:-4] for file in files]
    idx = 0
    for image, image_id in zip(imgs_mask_test, imgs_mask_test_gt):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, files[idx] + '_pred.png'), image)
        idx = idx + 1

    ev = model.evaluate(imgs_test, imgs_mask_test_gt)
    dice, iou = ev[1], ev[2]

    print("dice score:", dice)
    print("iou score:", iou)

    # l=[]
    # for i in range(len(imgs_test)):
    #    l.append(model.evaluate(imgs_test[i,:].reshape(1,256,256,3), imgs_id_test[i,:].reshape(1,256,256,1))[2])
    #
    # ll = [elt for elt in l if elt>=0.9]
    #
    # np.mean(ll)

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coefficients')
    plt.ylabel('dice coefficients')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()

    plt.plot(history.history['iou_coef'])
    plt.plot(history.history['val_iou_coef'])
    plt.title('model iou coefficients')
    plt.ylabel('iou coefficients')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(format = '%(process)d-%(levelname)s-%(message)s', level = logging.INFO)
    logger.info("start!")
    start_time = time.time()
    main()
    logger.info("--- %s seconds ---" % (time.time() - start_time))
