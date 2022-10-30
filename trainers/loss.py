import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend

# weighted_loss = tf.reshape(tf.constant([0.3,0.7]), [1, 1,1, 3])
def weighted_categorical_crossentropy(weights):
    
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
def SparseCategoricalCrossentropy(y_true,y_pred):
    y_true=tf.cast(y_true,tf.uint8)
    y_true=tf.one_hot(y_true, depth=3)#(13, 256, 256, 3)
    y_true=tf.reshape(y_true,y_pred.shape)
    return - tf.reduce_sum(y_true * tf.math.log(y_pred+1e-7),axis=-1)#/y_true.shape[0]


def dice_loss_2d(y_true,y_pred):
    probabilities =y_pred# tf.nn.softmax(y_pred)
    # print(y_true.shape)
    # print(tf.executing_eagerly())
    y_true=tf.cast(y_true,tf.uint8)
    y_true=tf.one_hot(y_true, depth=3)#(13, 256, 256, 3)
    y_true=tf.reshape(y_true,probabilities.shape)
    
    aa=tf.reduce_sum(y_true * probabilities, axis=(1, 2))
    print(y_true.shape,probabilities.shape,aa.shape,weighted_loss[0, 0, 0].shape)
    numerator = 2 * tf.reduce_sum(weighted_loss[0, 0, 0] *
                                    tf.reduce_sum(y_true * probabilities, axis=(1, 2)), axis=1)
    denominator = tf.reduce_sum(weighted_loss[0, 0, 0] *
                                tf.reduce_sum(y_true + probabilities, axis=(1, 2)), axis=1)


    loss = 1 - tf.reduce_mean(numerator / denominator)
    # print('{} / {}'.format(numerator, denominator))
    return loss
def dice_loss_with_Focal_Loss(cls_weights,
                              beta=1,
                              smooth=1e-5,
                              alpha=0.5,
                              gamma=2):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])

    def _dice_loss_with_Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        logpt = -y_true[..., :-1] * K.log(y_pred) * cls_weights
        logpt = -K.sum(logpt, axis=-1)

        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt)**gamma) * logpt
        CE_loss = K.mean(CE_loss)

        tp = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        fp = K.sum(y_pred, axis=[0, 1, 2]) - tp
        fn = K.sum(y_true[..., :-1], axis=[0, 1, 2]) - tp

        score = ((1 + beta**2) * tp + smooth) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss

    return _dice_loss_with_Focal_Loss

def Focal_Loss2(cls_weights, alpha=0.5, gamma=2):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        logpt = - y_true * K.log(y_pred)*cls_weights
        logpt = - K.sum(logpt, axis = -1)

        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)
        return CE_loss
    return _Focal_Loss

def Focal_Loss(cls_weights, alpha=0.5, gamma=2):
    #l=len(cls_weights)
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _Focal_Loss(y_true, y_pred):
        # print("pppp",cls_weights.shape,y_pred.shape)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # y_true=tf.cast(y_true,tf.uint8)
        # y_true=tf.one_hot(y_true, depth=l)#(13, 256, 256, 3)
        y_true=tf.cast(y_true,tf.float32)
        
        #y_true=tf.reshape(y_true,y_pred.shape)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return (-(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1+K.epsilon())) - ((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + K.epsilon())))*cls_weights
        
    return _Focal_Loss

def CE(cls_weights):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _CE(y_true, y_pred):
        y_true=tf.cast(y_true,tf.uint8)
        y_true=tf.one_hot(y_true, depth=3)#(13, 256, 256, 3)
        y_true=tf.reshape(y_true,y_pred.shape)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true * K.log(y_pred) * cls_weights
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE


if __name__=="__main__":
    y_true = tf.constant([1, 2])
    y_pred = tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    a=SparseCategoricalCrossentropy(y_true,y_pred)
    b=tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    print(a,b)