import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Add, Concatenate, Conv2D, Input, Lambda,
                                     Layer, Reshape, Softmax, UpSampling2D,
                                     ZeroPadding2D)

from nets.yolact_training import yolact_Loss



#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from tensorflow.keras import layers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Input, MaxPooling2D, ZeroPadding2D)


def identity_block(input_tensor, kernel_size, filters, name=""):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), name=name+'.conv1', use_bias=False)(input_tensor)
    # x = BatchNormalization(epsilon=1e-5, name=name+'.bn1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filters2, kernel_size, padding='valid', name=name+'.conv2', use_bias=False)(x)
    # x = BatchNormalization(epsilon=1e-5, name=name+'.bn2')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=name+'.conv3', use_bias=False)(x)
    # x = BatchNormalization(epsilon=1e-5, name=name+'.bn3')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2), name=""):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), name=name+'.conv1', use_bias=False)(input_tensor)
    # x = BatchNormalization(epsilon=1e-5, name=name+'.bn1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filters2, kernel_size, strides=strides, padding='valid', name=name+'.conv2', use_bias=False)(x)
    # x = BatchNormalization(epsilon=1e-5, name=name+'.bn2')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=name+'.conv3', use_bias=False)(x)
    # x = BatchNormalization(epsilon=1e-5, name=name+'.bn3')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=name+'.downsample.0', use_bias=False)(input_tensor)
    # shortcut = BatchNormalization(epsilon=1e-5, name=name+'.downsample.1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(inputs):
    # 544, 544, 3 -> 272, 272, 64
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='backbone.conv1',use_bias=False)(x)
    # x = BatchNormalization(epsilon=1e-5, name='backbone.bn1')(x)
    x = Activation('relu')(x)

    # 272, 272, 64 -> 136, 136, 64
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)

    # 136, 136, 64 -> 136, 136, 256
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1), name="backbone.layers.0.0")
    x = identity_block(x, 3, [64, 64, 256], name="backbone.layers.0.1")
    x = identity_block(x, 3, [64, 64, 256], name="backbone.layers.0.2")

    # 136, 136, 256 -> 68, 68, 512
    x = conv_block(x, 3, [128, 128, 512], name="backbone.layers.1.0")
    x = identity_block(x, 3, [128, 128, 512], name="backbone.layers.1.1")
    x = identity_block(x, 3, [128, 128, 512], name="backbone.layers.1.2")
    x = identity_block(x, 3, [128, 128, 512], name="backbone.layers.1.3")
    y1 = x

    # 68, 68, 512 -> 34, 34, 1024
    x = conv_block(x, 3, [256, 256, 1024], name="backbone.layers.2.0")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.1")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.2")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.3")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.4")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.5")
    y2 = x

    # 34, 34, 1024 -> 17, 17, 2048
    x = conv_block(x, 3, [512, 512, 2048], name="backbone.layers.3.0")
    x = identity_block(x, 3, [512, 512, 2048], name="backbone.layers.3.1")
    x = identity_block(x, 3, [512, 512, 2048], name="backbone.layers.3.2")
    y3 = x
    return y1, y2, y3

class UpsampleLike(Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class Upsample(Layer):
    def call(self, inputs, **kwargs):
        inputs_shape = keras.backend.shape(inputs)
        return tf.compat.v1.image.resize_images(inputs, (inputs_shape[1] * 2, inputs_shape[2] * 2), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[-1])

def FPN(C3,C4,C5):
    P5           = Conv2D(256, kernel_size=1, strides=1, padding='same', name='fpn.lat_layers.2')(C5)
    P4           = Conv2D(256, kernel_size=1, strides=1, padding='same', name='fpn.lat_layers.1')(C4)
    P3           = Conv2D(256, kernel_size=1, strides=1, padding='same', name='fpn.lat_layers.0')(C3)

    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, P4])
    P4           = Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, P3])
    P3           = Add(name='P3_merged')([P4_upsampled, P3])

    P5           = Conv2D(256, kernel_size=3, strides=1, padding='same', name='fpn.pred_layers.2.0', activation='relu')(P5)
    P4           = Conv2D(256, kernel_size=3, strides=1, padding='same', name='fpn.pred_layers.1.0', activation='relu')(P4)
    P3           = Conv2D(256, kernel_size=3, strides=1, padding='same', name='fpn.pred_layers.0.0', activation='relu')(P3)

    P6           = ZeroPadding2D((1, 1))(P5)
    P6           = Conv2D(256, kernel_size=3, strides=2, padding='valid', name='fpn.downsample_layers.0.0', activation='relu')(P6)
    P7           = ZeroPadding2D((1, 1))(P6)
    P7           = Conv2D(256, kernel_size=3, strides=2, padding='valid', name='fpn.downsample_layers.1.0', activation='relu')(P7)
    return P3, P4, P5, P6, P7

def Protonet(x, num_prototype):
    x = Conv2D(256, (3, 3), padding="same", name='proto_net.proto1.0', activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="same", name='proto_net.proto1.2', activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="same", name='proto_net.proto1.4', activation="relu")(x)
    x = Upsample()(x)
    x = Conv2D(256, (3, 3), padding="same", name='proto_net.proto2.0', activation="relu")(x)
    x = Conv2D(num_prototype, (1, 1), padding="same", name='proto_net.proto2.2', activation='relu')(x)
    return x

class PredictionModule():
    def __init__(self, out_channels, num_anchors, num_class, num_mask):
        super(PredictionModule, self).__init__()
        self.num_anchors    = num_anchors

        self.Conv           = Conv2D(out_channels, (3, 3), padding="same", name='prediction_layers.upfeature.0', activation="relu")

        self.boxConv        = Conv2D(self.num_anchors * 4, (3, 3), padding="same", name='prediction_layers.bbox_layer')
        self.boxReshape     = Reshape([-1, 4])

        self.classConv      = Conv2D(self.num_anchors * num_class, (3, 3), padding="same", name='prediction_layers.conf_layer')
        self.classReshape   = Reshape([-1, num_class])

        self.maskConv       = Conv2D(self.num_anchors * num_mask, (3, 3), padding="same", name='prediction_layers.coef_layer.0', activation='tanh')
        self.maskReshape    = Reshape([-1, num_mask])

    def call(self, p):
        p = self.Conv(p)

        pred_box    = self.boxConv(p)
        pred_box    = self.boxReshape(pred_box)

        pred_class  = self.classConv(p)
        pred_class  = self.classReshape(pred_class)

        pred_mask   = self.maskConv(p)
        pred_mask   = self.maskReshape(pred_mask)

        return [pred_box, pred_class, pred_mask]


def yolact(input_shape, num_classes, num_mask = 32, train_mode = False):
    inputs          = Input(shape=input_shape)
    #----------------------------#
    #   获得的C3为68, 68, 512
    #   获得的C4为34, 34, 1024
    #   获得的C5为17, 17, 2048
    #----------------------------#
    C3, C4, C5      = ResNet50(inputs)

    #----------------------------#
    #   获得的P3为68, 68, 256
    #   获得的P4为34, 34, 256
    #   获得的P5为17, 17, 256
    #   获得的P6为9, 9, 256
    #   获得的P7为5, 5, 256
    #----------------------------#
    features        = FPN(C3,C4,C5)

    pred_boxes      = []
    pred_classes    = []
    pred_masks      = []
    #--------------------------------------------#
    #   将5个特征层利用同一个head的预测结果堆叠
    #   pred_boxes      18525, 4
    #   pred_classes    18525, 81
    #   pred_masks      18525, 32
    #--------------------------------------------#
    predictionmodule = PredictionModule(256, 3, num_classes, num_mask)
    for f_map in features:
        box, cls, mask = predictionmodule.call(f_map)
        pred_boxes.append(box)
        pred_classes.append(cls)
        pred_masks.append(mask)
    pred_boxes      = Concatenate(axis = 1)(pred_boxes)
    pred_classes    = Concatenate(axis = 1)(pred_classes)
    pred_masks      = Concatenate(axis = 1)(pred_masks)

    #-------------------------------------#
    #   对P3进行上采样
    #   获得的pred_proto为136, 136, 32
    #-------------------------------------#
    pred_proto      = Protonet(features[0], num_mask)
    
    seg             = Conv2D(num_classes - 1, (1, 1), padding="same", activation='sigmoid', name='semantic_seg_conv')(features[0])
    if not train_mode:
        pred_classes = Softmax(axis = -1, name="softmax")(pred_classes)
    model           = keras.models.Model(inputs, [pred_boxes, pred_classes, pred_masks, pred_proto, seg])
    return model

def get_train_model(model_body):
    pred_boxes, pred_classes, pred_masks, pred_proto, seg = [*model_body.output]

    true_boxes          = Input(shape=[None, 4])
    true_classes        = Input(shape=[None])
    mask_gt             = Input(shape=[None, None, None])
    segment_gt          = Input(shape=[None, None, None])
    anchor_max_box      = Input(shape=[None, 4])
    anchor_max_index    = Input(shape=[None])

    loss_               = Lambda(yolact_Loss, name='yolact_Loss')(
        [pred_boxes, true_boxes, pred_classes, true_classes, pred_masks, pred_proto, mask_gt, seg, segment_gt, anchor_max_box, anchor_max_index]
    )

    model = keras.models.Model(inputs=[model_body.input, true_boxes, true_classes, mask_gt, segment_gt, anchor_max_box, anchor_max_index], outputs=[loss_])
    return model
