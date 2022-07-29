from __future__ import print_function

from keras import Model
from keras.backend import expand_dims
from keras.layers import DepthwiseConv2D
from keras.layers import Dropout
from keras.layers import Lambda
from keras.optimizers import Adam
from tensorflow._api.v2.compat.v1 import image
from keras.layers import Input, concatenate, Conv2D, Add, MaxPooling2D, Activation, Dense, Reshape, \
    GlobalAveragePooling2D, Multiply, Conv2DTranspose, BatchNormalization, ReLU, UpSampling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import preprocess_input

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth = 1):
    intersection = K.sum(K.abs(y_true * y_pred), axis = [1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis = 0)
    return iou


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(0.8 * K.pow((1 - BCE_EXP), 2.) * BCE)
    return focal_loss


def loss(y_true, y_pred):
    return -(0.4 * dice_coef(y_true, y_pred) + 0.6 * iou_coef(y_true, y_pred))


def ConvBNReLU(inputs,
               outputSize,
               filterSize,
               dropout = 0.0,
               strides = 1,
               batchNormalization = False,
               padding = 'same',
               activation = True,
               useBias = False, repeats = 1):
    # for _ in range(repeats):

    output = inputs

    for _ in range(repeats):

        output = Conv2D(filters = outputSize,
                        kernel_size = filterSize,
                        strides = strides,
                        padding = padding,
                        use_bias = useBias)(output)  # CONV2D

        if batchNormalization is True:
            output = BatchNormalization(axis = 3)(output)

        if activation:
            output = ReLU()(output)  # RELU

    if dropout > 0.0:
        output = Dropout(rate = dropout)(output)

    return output


def ResidualConvBlock(inputs,
                      outputSize,
                      filterSize,
                      dropout = 0,
                      strides = 1,
                      batchNormalization = False,
                      padding = 'same',
                      activation = True,
                      use_bias = False, repeat = 1):
    output = ConvBNReLU(inputs = inputs, outputSize = outputSize,
                        filterSize = filterSize,
                        dropout = dropout,
                        strides = strides,
                        batchNormalization = batchNormalization,
                        padding = padding,
                        activation = activation,
                        useBias = use_bias, repeats = repeat)

    shortcut = Conv2D(outputSize, kernel_size = (1, 1), padding = 'same')(inputs)

    if batchNormalization is True:
        shortcut = BatchNormalization(axis = 3)(shortcut)

    output = Add()([shortcut, output])
    output = SqueezeExciteBlock(inputs = output, inputShape = output.shape)
    return output


def SqueezeExciteBlock(inputs, inputShape, ratio = 8):
    ratio = ratio
    globalAveragePooling2D = GlobalAveragePooling2D()
    squeezeExciteShape = (1, 1, inputShape[-1])
    reshape = Reshape(squeezeExciteShape)
    reLuDenseLayer = Dense(inputShape[-1] // ratio, activation = 'relu',
                           kernel_initializer = 'he_normal',
                           use_bias = False)
    sigmoidDenseLayer = Dense(inputShape[-1],
                              activation = 'sigmoid',
                              kernel_initializer = 'he_normal',
                              use_bias = False)

    squeeze = globalAveragePooling2D(inputs)
    squeeze = reshape(squeeze)
    squeeze = reLuDenseLayer(squeeze)
    squeeze = sigmoidDenseLayer(squeeze)
    output = Multiply()([inputs, squeeze])

    return output


def ASPPBlock(inputs, filters, batchNormalization = True):
    globalAveragePooling2D = GlobalAveragePooling2D()

    inputShape = inputs.get_shape().as_list()
    output = globalAveragePooling2D(inputs)
    output = expand_dims(expand_dims(output, 1), 1)

    output = ConvBNReLU(
            inputs = output,
            outputSize = filters,
            filterSize = (1, 1),
            dropout = 0,
            strides = 1,
            batchNormalization = batchNormalization,
            padding = 'same',
            activation = True,
            useBias = not batchNormalization)

    output = Lambda(lambda i: image.resize(i, inputShape[1:3], method = 'bilinear', align_corners = True))(output)

    output0 = ConvBNReLU(
            inputs = inputs,
            outputSize = filters,
            filterSize = (1, 1),
            dropout = 0,
            strides = 1,
            batchNormalization = batchNormalization,
            padding = 'same',
            activation = True,
            useBias = not batchNormalization)

    # dilation rates are fixed to `[6, 9, 12]`.

    outputD6 = DepthWiseConvBNReLU(filters = filters,
                                   kernelSize = 3,
                                   dilationRate = 6,
                                   batchNormalization = True,
                                   inputs = inputs)
    outputD9 = DepthWiseConvBNReLU(filters = filters,
                                   kernelSize = 3,
                                   dilationRate = 12,
                                   batchNormalization = True,
                                   inputs = inputs)
    outputD12 = DepthWiseConvBNReLU(filters = filters,
                                    kernelSize = 3,
                                    dilationRate = 18,
                                    batchNormalization = True,
                                    inputs = inputs)

    out = concatenate([output, output0, outputD6, outputD9, outputD12])

    y = Conv2D(filters, (1, 1), padding = "same")(out)

    # out = Conv2DTranspose(1024, (2, 2), strides = (2, 2), padding = 'same')(y)
    # print(y.shape)
    return y


def DepthWiseConvBNReLU(inputs,
                        filters,
                        kernelSize = 3,
                        dilationRate = 1,
                        padding = 'same',
                        batchNormalization = False):
    useBias = not batchNormalization
    depthWiseConv2D = DepthwiseConv2D(kernel_size = kernelSize,
                                      dilation_rate = dilationRate,
                                      padding = padding,
                                      use_bias = useBias)

    output = depthWiseConv2D(inputs)

    if batchNormalization:
        output = BatchNormalization(axis = -1)(output)

    output = ReLU()(output)
    output = ConvBNReLU(inputs = output,
                        outputSize = filters,
                        filterSize = kernelSize,
                        dropout = 0,
                        strides = 1,
                        batchNormalization = batchNormalization,
                        padding = 'same',
                        activation = True,
                        useBias = not batchNormalization,
                        repeats = 1

                        )

    return output


def AttentionBlock(inputs,
                   gatingSignal,
                   interShape):
    inputShape = K.int_shape(inputs)
    gatingShape = K.int_shape(gatingSignal)

    # Getting the inputShape signal to the same shape as the gating signal
    thetaInput = Conv2D(filters = interShape, kernel_size = (2, 2), strides = (2, 2), padding = 'same')(
            inputs)  # 16
    shapeThetaInput = K.int_shape(thetaInput)

    # Getting the gating signal to the same number of filters as the inter_shape
    phiGating = Conv2D(filters = interShape, kernel_size = (1, 1), padding = 'same')(gatingSignal)

    upsampleGating = Conv2DTranspose(filters = interShape,
                                     kernel_size = (3, 3),
                                     strides = (
                                         shapeThetaInput[1] // gatingShape[1],
                                         shapeThetaInput[2] // gatingShape[2]),
                                     padding = 'same')(phiGating)  # 16

    concatInputAndGating = Add()([upsampleGating, thetaInput])

    activateConcatSignal = Activation(activation = 'relu')(concatInputAndGating)
    psi = Conv2D(1, (1, 1), padding = 'same')(activateConcatSignal)

    sigmoidConcatSignal = Activation(activation = 'sigmoid')(psi)
    shapeSigmoidConcat = K.int_shape(sigmoidConcatSignal)

    upsamplePsi = UpSampling2D(
            size = (inputShape[1] // shapeSigmoidConcat[1], inputShape[2] // shapeSigmoidConcat[2]))(
            sigmoidConcatSignal)  # 32

    upsamplePsi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis = 3),
                         arguments = {'repnum': inputShape[3]})(upsamplePsi)

    output = Multiply()([upsamplePsi, inputs])

    output = Conv2D(filters = inputShape[3], kernel_size = (1, 1), padding = 'same')(output)
    output = BatchNormalization()(output)
    return output


def EncoderBlock1(inputs, convFilterSize, numOfFilters, dropout, batchNormalization):
    conv256 = ResidualConvBlock(outputSize = numOfFilters,
                                filterSize = convFilterSize,
                                dropout = dropout,
                                batchNormalization = batchNormalization, inputs = inputs, repeat = 2)
    maxPooling128 = MaxPooling2D(pool_size = (2, 2))(conv256)

    conv128 = ResidualConvBlock(outputSize = numOfFilters * 2,
                                filterSize = convFilterSize,
                                dropout = dropout,
                                batchNormalization = batchNormalization, inputs = maxPooling128, repeat = 2)
    maxPooling64 = MaxPooling2D(pool_size = (2, 2))(conv128)

    conv64 = ResidualConvBlock(outputSize = numOfFilters * 4,
                               filterSize = convFilterSize,
                               dropout = dropout,
                               batchNormalization = batchNormalization, repeat = 2, inputs = maxPooling64)
    maxPooling32 = MaxPooling2D(pool_size = (2, 2))(conv64)

    conv32 = ResidualConvBlock(outputSize = numOfFilters * 8,
                               filterSize = convFilterSize,
                               dropout = dropout,
                               batchNormalization = batchNormalization, repeat = 2, inputs = maxPooling32)
    maxPooling16 = MaxPooling2D(pool_size = (2, 2))(conv32)

    # conv16 = residualConvBlock16(maxPooling16)

    skips = [conv256, conv128, conv64, conv32]

    return maxPooling16, skips


def EncoderBlock2(inputs, convFilterSize, numOfFilters, dropout, batchNormalization, decoderSkips):
    skip1 = decoderSkips.pop()
    conv256 = ResidualConvBlock(outputSize = numOfFilters,
                                filterSize = convFilterSize,
                                dropout = dropout,
                                batchNormalization = batchNormalization, inputs = inputs, repeat = 2)
    maxPooling128 = MaxPooling2D(pool_size = (2, 2))(conv256)
    concatenate128 = concatenate([Conv2D(filters = numOfFilters * 2, kernel_size = convFilterSize, activation = 'relu',
                                         padding = 'same')(maxPooling128), skip1])

    skip2 = decoderSkips.pop()
    conv128 = ResidualConvBlock(outputSize = numOfFilters * 2,
                                filterSize = convFilterSize,
                                dropout = dropout,
                                batchNormalization = batchNormalization, inputs = concatenate128, repeat = 2)
    maxPooling64 = MaxPooling2D(pool_size = (2, 2))(conv128)
    concatenate64 = concatenate([Conv2D(filters = numOfFilters * 4, kernel_size = convFilterSize, activation = 'relu',
                                        padding = 'same')(maxPooling64), skip2])

    skip3 = decoderSkips.pop()
    conv64 = ResidualConvBlock(outputSize = numOfFilters * 4,
                               filterSize = convFilterSize,
                               dropout = dropout,
                               batchNormalization = batchNormalization, inputs = concatenate64, repeat = 2)

    maxPooling32 = MaxPooling2D(pool_size = (2, 2))(conv64)
    concatenate32 = concatenate([Conv2D(filters = numOfFilters * 8, kernel_size = convFilterSize, activation = 'relu',
                                        padding = 'same')(maxPooling32), skip3])

    conv32 = ResidualConvBlock(outputSize = numOfFilters * 8,
                               filterSize = convFilterSize,
                               dropout = dropout,
                               batchNormalization = batchNormalization, inputs = concatenate32, repeat = 2)
    maxPooling16 = MaxPooling2D(pool_size = (2, 2))(conv32)

    encoderSkips = [conv256, conv128, conv64, conv32]

    return maxPooling16, encoderSkips


def DecoderBlock(inputs, convFilterSize, numOfFilters, upsamplingFiltersSize, dropout, batchNormalization,
                 encoderSkips):
    ##

    skip1 = encoderSkips.pop()

    gatingSignal32 = ConvBNReLU(outputSize = numOfFilters * 8,
                                filterSize = 1,
                                dropout = dropout,
                                batchNormalization = batchNormalization, inputs = inputs)

    attention32 = AttentionBlock(inputs = skip1,
                                 gatingSignal = gatingSignal32,
                                 interShape = numOfFilters * 8)
    upSample32 = UpSampling2D(size = upsamplingFiltersSize, data_format = "channels_last")(inputs)

    concat32 = concatenate([upSample32, attention32], axis = 3)

    resConv32 = ResidualConvBlock(outputSize = numOfFilters * 8,
                                  filterSize = convFilterSize,
                                  dropout = dropout,
                                  batchNormalization = batchNormalization, inputs = concat32)

    ###
    skip2 = encoderSkips.pop()
    gatingSignal64 = ConvBNReLU(outputSize = numOfFilters * 4,
                                filterSize = 1,
                                dropout = dropout,
                                batchNormalization = batchNormalization, inputs = resConv32)
    attention64 = AttentionBlock(inputs = skip2,
                                 gatingSignal = gatingSignal64,
                                 interShape = numOfFilters * 4)
    upSample64 = UpSampling2D(size = upsamplingFiltersSize, data_format = "channels_last")(resConv32)

    concat64 = concatenate([upSample64, attention64], axis = 3)
    resConv64 = ResidualConvBlock(outputSize = numOfFilters * 4,
                                  filterSize = convFilterSize,
                                  dropout = dropout,
                                  batchNormalization = batchNormalization, inputs = concat64, repeat = 2)

    ###
    skip3 = encoderSkips.pop()
    gatingSignal128 = ConvBNReLU(outputSize = numOfFilters * 2,
                                 filterSize = 1,
                                 dropout = dropout,
                                 batchNormalization = batchNormalization, inputs = resConv64)
    attention128 = AttentionBlock(inputs = skip3,
                                  gatingSignal = gatingSignal128,
                                  interShape = numOfFilters * 2)
    upSample128 = UpSampling2D(size = upsamplingFiltersSize, data_format = "channels_last")(resConv64)

    concat128 = concatenate([upSample128, attention128], axis = 3)
    resConv128 = ResidualConvBlock(outputSize = numOfFilters * 2,
                                   filterSize = convFilterSize,
                                   dropout = dropout,
                                   batchNormalization = batchNormalization, inputs = concat128, repeat = 2)
    ###

    ###
    skip4 = encoderSkips.pop()
    gatingSignal256 = ConvBNReLU(outputSize = numOfFilters,
                                 filterSize = 1,
                                 dropout = dropout,
                                 batchNormalization = batchNormalization, inputs = resConv128)
    attention256 = AttentionBlock(inputs = skip4,
                                  gatingSignal = gatingSignal256,
                                  interShape = numOfFilters * 2)
    upSample256 = UpSampling2D(size = upsamplingFiltersSize, data_format = "channels_last")(resConv128)

    concat256 = concatenate([upSample256, attention256], axis = 3)
    resConv256 = ResidualConvBlock(outputSize = numOfFilters,
                                   filterSize = convFilterSize,
                                   dropout = dropout,
                                   batchNormalization = batchNormalization, inputs = concat256, repeat = 2)
    ###

    decoderSkips = [resConv32, resConv64, resConv128]

    return resConv256, decoderSkips


def AttentionResUNet(inputShape, numClasses = 1, dropoutRate = 0.2, batchNormalization = True):
    convFilterSize = 3
    numOfFilters = 64
    upSamplingFiltersSize = 2

    inputs = Input(shape = inputShape, dtype = tf.float32)
    inputs = Lambda(lambda x: x / 255.)(inputs)

    encoded1, encoderSkips1 = EncoderBlock1(convFilterSize = convFilterSize,
                                            numOfFilters = numOfFilters,
                                            dropout = dropoutRate,
                                            batchNormalization = batchNormalization, inputs = inputs)

    # aspp1 = ASPPBlock(filters = 512, batchNormalization = batchNormalization, inputs = encoded1)

    decoded1, decoderSkips1 = DecoderBlock(convFilterSize = convFilterSize,
                                           numOfFilters = numOfFilters,
                                           upsamplingFiltersSize = upSamplingFiltersSize,
                                           dropout = dropoutRate,
                                           batchNormalization = batchNormalization, inputs = encoded1,
                                           encoderSkips = encoderSkips1)

    newInput = concatenate(
            [Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same')(decoded1), decoded1],
            axis = 3)

    encoded2, encoderSkips2 = EncoderBlock2(convFilterSize = convFilterSize,
                                            numOfFilters = numOfFilters,
                                            dropout = dropoutRate,
                                            batchNormalization = batchNormalization, inputs = newInput,
                                            decoderSkips = decoderSkips1)

    decoded2, decoderSkips2 = DecoderBlock(convFilterSize = convFilterSize,
                                           numOfFilters = numOfFilters,
                                           upsamplingFiltersSize = upSamplingFiltersSize,
                                           dropout = dropoutRate,
                                           batchNormalization = batchNormalization, inputs = encoded2,
                                           encoderSkips = encoderSkips2)

    # aspp2 = ASPPBlock(filters = 32, batchNormalization = batchNormalization, inputs = decoded2)

    segmented = Conv2D(filters = numClasses, kernel_size = (1, 1))(decoded2)
    segmented = BatchNormalization(axis = 3)(segmented)
    segmented = Activation('sigmoid')(segmented)

    model = Model(inputs = [inputs], outputs = [segmented])
    model.compile(optimizer = Adam(1e-4), loss = [loss], metrics = [dice_coef, iou_coef])
    return model


img_rows = 256
img_cols = 256

narr = np.load("/splits/Dataset-split-arrays/Testing-Final-IMG-Arrays/testing_img_array__3.npy")

narr = preprocess_input(narr)
model = AttentionResUNet((img_rows, img_cols, 3))
model.load_weights("/Users/pablo/Downloads/weights-improvement-{epoch_02d}-{val_loss_.2f}.hdf5")
a = model.predict(narr)

import matplotlib.pyplot as plt
plt.imshow(a)
plt.show()
