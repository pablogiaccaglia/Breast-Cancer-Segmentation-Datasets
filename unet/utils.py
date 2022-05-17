from keras import backend as K
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D

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

def aspp_block(x, num_filters, rate_scale = 1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate = (6 * rate_scale, 6 * rate_scale), padding = "same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate = (12 * rate_scale, 12 * rate_scale), padding = "same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate = (18 * rate_scale, 18 * rate_scale), padding = "same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding = "same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding = "same")(y)
    return y