import math

from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, \
    Conv2DTranspose, BatchNormalization, ReLU, UpSampling2D, LayerNormalization
from keras import Sequential
import tensorflow as tf
from keras.activations import sigmoid
from keras import Input
import keras.backend as K
import numpy as np
from keras.layers import Layer

smooth = 1.

print(f"Using tensorflow {tf.__version__}")  # make sure it's the nightly build

print(f"Using numpy {np.__version__}")

import tensorflow as tf


class ConvBNReLU(Layer):

    def __init__(self,
                 channelIn,
                 channelOut,
                 kernelSize,
                 strides = 1,
                 padding = 'valid',
                 activation = True):
        super(ConvBNReLU, self).__init__()

        self.conv = Conv2D(filters = channelOut,
                           kernel_size = kernelSize,
                           strides = strides,
                           padding = padding,
                           use_bias = False,
                           data_format = 'channels_first')
        self.batchNormalization = BatchNormalization()
        self.reLu = ReLU()
        self.activation = activation

    def call(self, inputs, *args, **kwargs):
        # inputs = tf.transpose(a = inputs, perm = [0, 3, 2, 1])

        output = self.conv(inputs)

        # output = tf.transpose(a = output, perm = [0, 3, 2, 1])

        output = self.batchNormalization(output)

        if self.activation:
            output = self.reLu(output)

        return output


class DoubleConv(Layer):

    def __init__(self, channelIn, channelOut):
        super(DoubleConv, self).__init__()

        self.conv = Sequential([
            ConvBNReLU(channelIn = channelIn,
                       channelOut = channelOut,
                       kernelSize = (3, 3),
                       strides = 1,
                       padding = "same"),

            ConvBNReLU(channelIn = channelIn,
                       channelOut = channelOut,
                       kernelSize = (3, 3),
                       strides = 1,
                       padding = "same",
                       activation = False)])

        self.conv1 = Conv2D(filters = channelOut, kernel_size = 1, data_format = 'channels_first')
        self.reLu = ReLU()
        self.batchNormalization = BatchNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        h = x
        x = self.conv1(x)

        x = self.batchNormalization(x)

        x = h + x
        x = self.reLu(x)
        return x


class UNetEncoder(Layer):

    def __init__(self):
        super(UNetEncoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = MaxPooling2D(pool_size = (2, 2), data_format = "channels_first")
        self.res2 = DoubleConv(64, 128)
        self.pool2 = MaxPooling2D(pool_size = (2, 2), data_format = "channels_first")
        self.res3 = DoubleConv(128, 256)
        self.pool3 = MaxPooling2D(pool_size = (2, 2), data_format = "channels_first")

    def call(self, inputs, *args, **kwargs):
        features = []
        x = self.res1(inputs)
        features.append(x)  # (224, 224, 64)

        x = self.pool1(x)

        x = self.res2(x)

        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)

        return x, features


class UNetDecoder(Layer):

    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.trans1 = Conv2DTranspose(256, (2, 2), strides = (2, 2), data_format = 'channels_first')
        self.res1 = DoubleConv(512, 256)
        self.trans2 = Conv2DTranspose(128, (2, 2), strides = (2, 2), data_format = 'channels_first')
        self.res2 = DoubleConv(256, 128)
        self.trans3 = Conv2DTranspose(64, (2, 2), strides = (2, 2), data_format = 'channels_first')
        self.res3 = DoubleConv(128, 64)

    # def forward(self, x, feature):
    def call(self, inputs, *args, **kwargs):
        feature = kwargs['features']

        x = self.trans1(inputs)  # (56, 56, 256)
        x = tf.concat(values = [feature[2], x], axis = 1)
        x = self.res1(x)  # (56, 56, 256)
        x = self.trans2(x)  # (112, 112, 128)
        x = tf.concat(values = [feature[1], x], axis = 1)
        x = self.res2(x)  # (112, 112, 128)
        x = self.trans3(x)  # (224, 224, 64)
        x = tf.concat(values = [feature[0], x], axis = 1)
        x = self.res3(x)
        return x


class ModuleExternalAttention(Layer):

    def __init__(self, dim, configs):
        super(ModuleExternalAttention, self).__init__()
        self.numOfHeads = configs["head"]
        self.coefficients = 4
        self.queryLiner = Dense(units = dim * self.coefficients, activation = 'linear', use_bias = True)
        self.numOfHeads = self.coefficients * self.numOfHeads
        self.k = 256 // self.coefficients
        self.denseLayer0 = Dense(self.k, activation = 'linear', use_bias = True)
        self.denseLayer1 = Dense(dim * self.coefficients // self.numOfHeads, activation = 'linear', use_bias = True)
        self.proj = Dense(dim, activation = 'linear', use_bias = True)

    def call(self, inputs, *args, **kwargs):
        B, N, C = inputs.shape
        x = self.queryLiner(inputs)
        x = tf.transpose(a = tf.reshape(tensor = x, shape = [B, N, self.numOfHeads, -1]),
                         perm = [0, 2, 1, 3])  # (1, 32, 225, 32)

        attn = self.denseLayer0(x)
        attn = sigmoid(axis = -2)(attn)
        attn = attn / (1e-9 + tf.reduce_sum(input_tensor = attn, axis = -1, keepdims = True))

        x = tf.reshape(tensor = tf.transpose(a = self.denseLayer1(attn), perm = [0, 2, 1, 3]), shape = [B, N, -1])

        x = self.proj(x)

        return x


class Attention(Layer):

    def __init__(self, dim, configs, axial = False):
        super(Attention, self).__init__()
        self.axial = axial
        self.dim = dim
        self.numOfHeads = configs["head"]
        self.attentionHeadSize = int(self.dim / configs["head"])
        self.allHeadSize = self.numOfHeads * self.attentionHeadSize

        self.queryLayer = Dense(self.allHeadSize, activation = 'linear', use_bias = True)
        self.keyLayer = Dense(self.allHeadSize, activation = 'linear', use_bias = True)
        self.valueLayer = Dense(self.allHeadSize, activation = 'linear', use_bias = True)

        self.out = Dense(self.dim, activation = 'linear', use_bias = True)
        self.sigmoid = sigmoid

    def transposeForScores(self, x):

        newXShape = x.shape[:-1] + (self.numOfHeads, self.attentionHeadSize)
        x = tf.reshape(tensor = x, shape = [*newXShape])
        return x

    def call(self, inputs, *args, **kwargs):
        # first row and col attention
        if self.axial:
            # row attention (single head attention)
            b, h, w, c = inputs.shape

            mixedQueryLayer = self.queryLayer(inputs)
            mixedKeyLayer = self.keyLayer(inputs)
            mixedValueLayer = self.valueLayer(inputs)

            queryLayerX = tf.reshape(tensor = mixedQueryLayer, shape = [b * h, w, -1])

            # mixedKeyLayer = tf.reshape(tensor = mixedKeyLayer, shape = [b * h, w, -1])

            keyLayerX = tf.transpose(a = tf.reshape(tensor = mixedKeyLayer, shape = [b * h, w, -1]), perm = [0, 2, 1])

            attentionScoresX = tf.matmul(queryLayerX,
                                         keyLayerX)  # (b*h, w, w)

            attentionScoresX = tf.reshape(tensor = attentionScoresX, shape = [b, -1, w,
                                                                              w])  # (b, h, w, w)

            # col attention  (single head attention)
            queryLayerY = tf.reshape(tensor = tf.transpose(a = mixedQueryLayer, perm = [0, 2, 1,
                                                                                        3]), shape = [
                b * w, h, -1])

            keyLayerY = tf.transpose(a = tf.reshape(tensor = tf.transpose(a = mixedKeyLayer, perm = [
                0, 2, 1, 3]), shape = [b * w, h, -1]), perm = [0, 2, 1])

            attentionScoreY = tf.matmul(queryLayerY,
                                        keyLayerY)  # (b*w, h, h)

            attentionScoreY = tf.reshape(tensor = attentionScoreY, shape = [b, -1, h,
                                                                            h])  # (b, w, h, h)

            return attentionScoresX, attentionScoreY, mixedValueLayer

        else:

            mixedQueryLayer = self.queryLayer(inputs)
            mixedKeyLayer = self.keyLayer(inputs)
            mixedValueLayer = self.valueLayer(inputs)

            queryLayer = tf.transpose(a = self.transposeForScores(mixedQueryLayer), perm = [
                0, 1, 2, 4, 3, 5])  # (b, p, p, head, n, c)
            keyLayer = tf.transpose(a = self.transposeForScores(mixedKeyLayer), perm = [
                0, 1, 2, 4, 3, 5])
            valueLayer = tf.transpose(a = self.transposeForScores(mixedValueLayer), perm = [
                0, 1, 2, 4, 3, 5])

            v1 = queryLayer

            v2 = tf.transpose(a = keyLayer, perm = [0, 1, 2, 3, 5, 4])

            attentionScores = tf.matmul(v1, v2)

            attentionScores = attentionScores / math.sqrt(
                    self.attentionHeadSize)
            attentionProbabilities = sigmoid(attentionScores, axis = -1)

            contextLayer = tf.matmul(
                    attentionProbabilities,
                    valueLayer)  # (b, p, p, head, win, h)

            contextLayer = tf.transpose(a = contextLayer, perm = [0, 1, 2, 4, 3, 5])

            newContextLayerShape = contextLayer.shape[:-2] + (
                self.allHeadSize,)
            contextLayer = tf.reshape(tensor = contextLayer, shape = [*newContextLayerShape])
            attentionOutput = self.out(contextLayer)

        return attentionOutput


def replace(original, toReplace):
    paddingLen1 = original.shape[2] - toReplace.shape[2]
    paddingLen2 = original.shape[3] - toReplace.shape[3]

    paddings1 = tf.constant([[0, 0], [0, 0], [0, paddingLen1], [0, paddingLen2]])

    mask = tf.zeros_like(toReplace)  # MASK

    yPadded = tf.pad(tensor = toReplace, paddings = paddings1, mode = "CONSTANT", constant_values = 0)
    qPadded = tf.pad(tensor = mask, paddings = paddings1, mode = "CONSTANT", constant_values = 1)

    a = original * qPadded + yPadded

    return a


def replace2(original, toReplace, delta):
    paddingLen1 = abs(toReplace.shape[2] - delta)
    paddingLen2 = abs(toReplace.shape[3] - delta)

    paddings1 = tf.constant([[0, 0], [0, 0], [paddingLen1, 0], [paddingLen2, 0]])

    mask = tf.zeros_like(toReplace)
    yPadded = tf.pad(tensor = toReplace, paddings = paddings1, mode = "CONSTANT", constant_values = 0)
    qPadded = tf.pad(tensor = mask, paddings = paddings1, mode = "CONSTANT", constant_values = 1)

    a = original * qPadded + yPadded

    return a


class WindowAttention(Layer):

    def __init__(self, configs, dim):
        super(WindowAttention, self).__init__()
        self.windowSize = configs["win_size"]
        self.attention = Attention(dim, configs)

    def call(self, inputs, *args, **kwargs):
        b, n, c = inputs.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = tf.reshape(tensor = tf.transpose(a = inputs, perm = [0, 2, 1]), shape = [b, c, h, w])

        if h % self.windowSize != 0:
            # print(x.shape)

            right_size = h + self.windowSize - h % self.windowSize
            newX = tf.zeros((b, c, right_size, right_size))

            # print(newX.shape)

            # print(x.shape[2] - right_size)
            # print(x.shape[3] - right_size)

            newX = replace(newX, x)
            newX = replace2(newX, x[:, :, (x.shape[2] - right_size):,
                                  (x.shape[3] - right_size):], right_size)

            x = newX

            b, c, h, w = x.shape

        x = tf.reshape(tensor = x,
                       shape = [b, c, h // self.windowSize, self.windowSize, w // self.windowSize, self.windowSize])
        x = tf.reshape(tensor = tf.transpose(a = x, perm = [0, 2, 4, 3, 5, 1]), shape = [b,
                                                                                         h // self.windowSize,
                                                                                         w // self.windowSize,
                                                                                         self.windowSize * self.windowSize,
                                                                                         c])
        x = self.attention(x)  # (b, p, p, win, h)
        return x


class LightDynamicConv(Layer):
    def __init__(self, configs):
        super(LightDynamicConv, self).__init__()
        self.denseLayer = Dense(configs["win_size"] * configs["win_size"])
        self.softmax = SoftMax(axis = -1)

    def call(self, inputs, *args, **kwargs):
        h = inputs

        avgX = tf.math.reduce_mean(inputs, axis = -2)  # (b, n, n, h)

        xProb = self.sigmoid(self.denseLayer(avgX))  # (b, n, n, win)

        x = tf.math.multiply(h, tf.expand_dims(xProb, axis = -1)
                             )  # (b, p, p, 16, h) (b, p, p, 16)

        x = tf.math.reduce_sum(x, axis = -2)  # (b, n, n, h)

        return x


class GaussianTrans(Layer):
    def __init__(self):
        super(GaussianTrans, self).__init__()

        self.bias = tf.convert_to_tensor(tf.Variable(initial_value = -tf.math.abs(x = tf.random.normal(shape = [1]))))
        self.shift = tf.convert_to_tensor(tf.Variable(initial_value = tf.math.abs(x = tf.random.normal(shape = [1]))))
        self.sigmoid = sigmoid(axis = -1)

    def call(self, inputs, *args, **kwargs):
        x, attentionXFull, attentionYFull, valueFull = inputs  # attentionXFull(b, h, w, c)   attentionYFull(b, w, h, c) valueFull(b, h, w, c)

        newValueFull = tf.zeros_like(input = valueFull)

        for r in range(x.shape[1]):  # row
            for c in range(x.shape[2]):  # col

                attentionX = attentionXFull[:, r, c, :]  # (b, w)
                attentionY = attentionYFull[:, c, r, :]  # (b, h)

                disX = tf.convert_to_tensor([(h - c) ** 2 for h in range(x.shape[2])
                                             ])  # (b, w)

                disY = tf.convert_to_tensor([(w - r) ** 2 for w in range(x.shape[1])
                                             ])

                disX = -(self.shift * tf.cast(disX, tf.float32) + self.bias)
                disY = -(self.shift * tf.cast(disY, tf.float32) + self.bias)

                attentionX = self.sigmoid(disX + attentionX)
                attentionY = self.sigmoid(disY + attentionY)

                attentionXExp = tf.expand_dims(attentionX, axis = -1)
                attentionYExp = tf.expand_dims(attentionY, axis = -1)

                arr = np.ones(newValueFull.shape, dtype = np.float32)

                arr[:, r, c, :] = 0

                arr = tf.convert_to_tensor(arr)

                s = tf.reduce_sum(
                        input_tensor = attentionXExp * valueFull[:, r, :, :] + attentionYExp * valueFull[:, :, c, :],
                        axis = -2)

                newValueFull = newValueFull * arr + s * (1 - arr)

        return newValueFull


class CSAttention(Layer):

    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        self.winAttention = WindowAttention(configs, dim)
        self.dLightConv = LightDynamicConv(configs)
        self.globalAttention = Attention(dim, configs, axial = True)
        self.gaussianTrans = GaussianTrans()
        # self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        # self.maxpool = nn.MaxPool2d(2)
        self.up = UpSampling2D(size = 4, data_format = "channels_first", interpolation = 'bilinear')
        self.squeeze = Conv2D(filters = dim, kernel_size = 1, data_format = 'channels_first')
        self.dim = dim

    def call(self, inputs, *args, **kwargs):
        '''

        :param inputs: size(b, n, c)
        :return:
        '''

        origin_size = inputs.shape
        _, origin_h, origin_w, _ = origin_size[0], int(np.sqrt(
                origin_size[1])), int(np.sqrt(origin_size[1])), origin_size[2]
        x = self.winAttention(inputs)  # (b, p, p, win, h)

        b, p, p, win, c = x.shape
        h = tf.transpose(a = tf.reshape(tensor = x, shape = [b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
                                                             c]), perm = [0, 1, 3, 2, 4, 5])

        h = tf.transpose(a = tf.reshape(tensor = h, shape = [b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
                                                             c]), perm = [0, 3, 1, 2])  # (b, c, h', w')

        x = self.dLightConv(x)  # (b, n, n, h)

        attenX, attenY, mixed_value = self.globalAttention(
                x)  # (attenX, attenY, value)

        gaussianInput = (x, attenX, attenY, mixed_value)

        x = self.gaussianTrans(gaussianInput)  # (b, h, w, c)

        x = tf.transpose(a = x, perm = [0, 3, 1, 2])

        x = self.up(x)

        intr = tf.concat(values = [x, h], axis = 1)

        t = self.squeeze(intr)

        x = tf.transpose(a = t, perm = [0, 2, 3, 1])

        x = x[:, :origin_h, :origin_w, :]

        x = tf.reshape(tensor = x, shape = [b, -1, c])

        return x


class MTModule(Layer):
    def __init__(self, dim):
        super(MTModule, self).__init__()
        self.SelfLayerNorm = LayerNormalization(epsilon = 1e-6)
        self.ExternalLayerNorm = LayerNormalization(epsilon = 1e-6)
        self.CSAttention = CSAttention(dim, configs)
        self.EAttention = ModuleExternalAttention(dim, configs)

    def call(self, inputs, *args, **kwargs):
        h = inputs  # (B, N, H)

        x = self.SelfLayerNorm(inputs)

        x = self.CSAttention(x)  # padding right_size

        x = h + x

        h = x
        x = self.ExternalLayerNorm(x)

        x = self.EAttention(x)
        x = h + x

        return x


class DecoderStem(Layer):
    def __init__(self):
        super(DecoderStem, self).__init__()
        self.block = UNetDecoder()

    def call(self, inputs, *args, **kwargs):
        features = kwargs['features']
        x = self.block(inputs = inputs, features = features)
        return x


class Stem(Layer):
    def __init__(self):
        super(Stem, self).__init__()
        self.model = UNetEncoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 'valid')  # out_dim, model_dim
        self.position_embedding = tf.Variable(tf.zeros(shape = (1, 784, 256)))

    def call(self, inputs, *args, **kwargs):
        x, features = self.model(inputs = inputs)  # (1, 512, 28, 28)

        x = self.trans_dim(inputs = x)  # (B, C, H, W) (1, 256, 28, 28)

        x = tf.reshape(x, shape = (1, 256, 28 * 28))  # (B, H, N)  (1, 256, 28*28)

        x = tf.transpose(x, perm = [0, 2, 1])  # (B, N, H)

        x = x + self.position_embedding
        return x, features  # (B, N, H)


class EncoderBlock(Layer):
    def __init__(self, dim):
        super(EncoderBlock, self).__init__()
        self.block = [
            MTModule(dim),
            MTModule(dim),
            ConvBNReLU(dim, dim * 2, 2, strides = 2, padding = 'valid')
        ]

    def call(self, inputs, *args, **kwargs):
        x = self.block[0](inputs)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = tf.transpose(a = tf.reshape(tensor = x, shape = [B, h, w, C]), perm = [0, 3, 1,
                                                                                   2])  # (1, 256, 28, 28) B, C, H, W
        skip = x
        x = self.block[2](x)  # (14, 14, 256)
        return x, skip


class DecoderBlock(Layer):

    def __init__(self, dim, flag):
        super(DecoderBlock, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = [
                Conv2DTranspose(filters = dim // 2,
                                kernel_size = 2,
                                strides = 2,
                                padding = 'valid', data_format = 'channels_first'),
                Conv2D(filters = dim // 2, kernel_size = 1, strides = 1, data_format = 'channels_first'),
                MTModule(dim // 2),
                MTModule(dim // 2)
            ]
        else:
            self.block = [
                Conv2DTranspose(filters = dim // 2,
                                kernel_size = 2,
                                strides = 2,
                                padding = 'valid', data_format = 'channels_first'),
                MTModule(dim),
                MTModule(dim)
            ]

    def call(self, inputs, *args, **kwargs):
        skips = kwargs['skips']

        if not self.flag:
            x = self.block[0](inputs)
            x = tf.concat(values = [x, skips], axis = 1)
            x = self.block[1](x)
            x = tf.transpose(a = x, perm = [0, 2, 3, 1])
            B, H, W, C = x.shape
            x = tf.reshape(tensor = x, shape = [B, -1, C])
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](inputs)
            x = tf.concat(values = [x, skips], axis = 1)
            x = tf.transpose(a = x, perm = [0, 2, 3, 1])
            B, H, W, C = x.shape
            x = tf.reshape(tensor = x, shape = [B, -1, C])
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class MTUNet(Layer):

    def __init__(self, outChannels = 4):
        super(MTUNet, self).__init__()
        self.stem = Stem()
        self.encoder = []
        self.bottleneck = Sequential([MTModule(configs["bottleneck"]),
                                      MTModule(configs["bottleneck"])])
        self.decoder = []

        self.decoderStem = DecoderStem()
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(EncoderBlock(dim))
        for i in range(len(configs["decoder"]) - 1):
            dim = configs["decoder"][i]
            self.decoder.append(DecoderBlock(dim, False))
        self.decoder.append(DecoderBlock(configs["decoder"][-1], True))
        self.SegmentationHead = Conv2D(filters = outChannels, kernel_size = 1, data_format = 'channels_first')

    def call(self, inputs, *args, **kwargs):

        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        x, features = self.stem(inputs)  # (B, N, C) (1, 196, 256)

        skips = []

        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  # (1, 512, 8, 8)
            x = tf.reshape(tensor = tf.transpose(a = x, perm = [0, 2, 3, 1]), shape = [B, -1, C])  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        B, N, C = x.shape
        x = tf.transpose(a = tf.reshape(tensor = x, shape = [B, int(np.sqrt(N)), -1, C]), perm = [0, 3, 1, 2])

        for i in range(len(self.decoder)):
            x = self.decoder[i](x,
                                skips = skips[len(self.decoder) - i - 1])  # (B, N, C)
            B, N, C = x.shape

            x = tf.transpose(a = tf.reshape(tensor = x, shape = [B, int(np.sqrt(N)), int(np.sqrt(N)),
                                                                 C]), perm = [0, 3, 1, 2])

        x = self.decoderStem(x, features = features)
        x = self.SegmentationHead(x)
        return x


def buildMTUnet(inputs):
    inputs = inputs

    stem = Stem()
    encoder = []
    bottleneck = Sequential([MTModule(configs["bottleneck"]),
                             MTModule(configs["bottleneck"])])
    decoder = []

    decoderStem = DecoderStem()
    for i in range(len(configs["encoder"])):
        dim = configs["encoder"][i]
        encoder.append(EncoderBlock(dim))
    for i in range(len(configs["decoder"]) - 1):
        dim = configs["decoder"][i]
        decoder.append(DecoderBlock(dim, False))
    decoder.append(DecoderBlock(configs["decoder"][-1], True))
    SegmentationHead = Conv2D(filters = 1, kernel_size = 1, data_format = 'channels_first')

    inputs = tf.transpose(inputs, [0, 3, 1, 2])

    if inputs.shape[1] == 1:
        inputs = inputs.repeat(1, 3, 1, 1)

    x, features = stem(inputs)  # (B, N, C) (1, 196, 256)

    skips = []

    for i in range(len(encoder)):
        x, skip = encoder[i](x)
        skips.append(skip)
        B, C, H, W = x.shape  # (1, 512, 8, 8)
        x = tf.reshape(tensor = tf.transpose(a = x, perm = [0, 2, 3, 1]), shape = [B, -1, C])  # (B, N, C)
    x = bottleneck(x)  # (1, 25, 1024)
    B, N, C = x.shape
    x = tf.transpose(a = tf.reshape(tensor = x, shape = [B, int(np.sqrt(N)), -1, C]), perm = [0, 3, 1, 2])

    for i in range(len(decoder)):
        x = decoder[i](x,
                       skips = skips[len(decoder) - i - 1])  # (B, N, C)
        B, N, C = x.shape

        x = tf.transpose(a = tf.reshape(tensor = x, shape = [B, int(np.sqrt(N)), int(np.sqrt(N)),
                                                             C]), perm = [0, 3, 1, 2])

    x = decoderStem(x, features = features)
    x = SegmentationHead(x)
    return x


configs = {
    "win_size":    4,
    "head":        8,
    "axis":        [28, 16, 8],
    "encoder":     [256, 512],
    "bottleneck":  1024,
    "decoder":     [1024, 512],
    "decoderStem": [(256, 512), (256, 256), (128, 64), 32]
}


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


inputs = Input(shape = (224,224,3), batch_size = None)

i = tf.zeros([2, 2])

# a = tf.Variable(inputs)

i = abs(np.random.rand(1, 224, 224, 3))

i = i / i.max()

i = np.transpose(a = i, axes = [0, 3, 1, 2])

out = buildMTUnet(inputs)

print(out)

model = Model(inputs = [inputs], outputs = [out])
model.summary()
