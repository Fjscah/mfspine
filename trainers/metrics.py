import csv
import os
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import backend
import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice




def Iou_score(smooth = 1e-5, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())
        
        intersection = backend.sum(y_true * y_pred, axis=[0,1,2])
        union = backend.sum(y_true + y_pred, axis=[0,1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score
    return _Iou_score

class MeanIoU(tf.keras.metrics.MeanIoU):
    """
    y_true: Tensor，真实标签（one-hot类型），
    y_pred: Tensor，模型输出结果（one-hot类型），二者shape都为[N,H,W,C],C为总类别数,
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
        return self.result()

def MIou_score(cls_weights,smooth = 1e-5, threhold = 0.5):
    #error
    l=len(cls_weights)
    def _MIou(y_true, y_pred):
        # score calculation
        y_pred = K.clip(y_pred, 0, 1.0 - K.epsilon())
        #y_true=tf.cast(y_true,tf.uint8)
        #y_true=tf.one_hot(y_true[...,0], depth=l)#(13, 256, 256, 3)
        y_true=tf.cast(y_true,tf.float32)
        ysum=y_pred+y_true
        ysum = K.clip(ysum, 0, 1.0 - K.epsilon())
        inter = tf.reduce_sum((y_true * y_pred),axis=[0,1,2])
        union = tf.reduce_sum(ysum,axis=[0,1,2])
        # print(union.shape)
        score = ((inter + smooth) / (union + smooth))[1:]
        return score
    return _MIou

def cal_mean_iou(num_classes, ignore_labels=None):
    """
    num_classes: int, 表示类别总数
    ignore_labels: list[int]，注意这里ignore_labels必须为列表或None，
    若为列表则存放int类型数字，表示需要忽略（不需要计算miou）的类别，
    例如：num_classes=12 ignore_labels=[11] 表示总类别数为12，忽略第11个类别
    """
    
    def MIOU(y_true, y_pred):
        """
        y_true: Tensor，真实标签（one-hot类型），
        y_pred: Tensor，模型输出结果（one-hot类型），二者shape都为[N,H,W,C]或[N,H*W,C],C为总类别数,
        """
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])  # 求argmax后，展平为一维
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
        num_need_labels = num_classes #不忽略的类别总数
        if ignore_labels is not None:
            num_need_labels -= len(ignore_labels)
            for ig in ignore_labels:
                mask = tf.not_equal(y_true, ignore_labels)  # 获取需要忽略的标签的位置
                y_true = tf.boolean_mask(y_true, mask)  # 剔除y_true中需要忽略的标签
                y_pred = tf.boolean_mask(y_pred, mask)  # 剔除y_pred中需要忽略的标签
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes)  # 计算混淆矩阵
        intersect = tf.linalg.diag_part(confusion_matrix)  # 获取对角线上的矩阵，形成一维向量
        union = tf.reduce_sum(confusion_matrix, axis=0) + tf.reduce_sum(confusion_matrix, axis=1) - intersect
        iou = tf.math.divide_no_nan(tf.cast(intersect, tf.float32), tf.cast(union, tf.float32))
        num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(union, 0), dtype=tf.float32)) #统计union中不为0的总数
        num = tf.minimum(num_valid_entries, num_need_labels)
        mean_iou = tf.math.divide_no_nan(tf.reduce_sum(iou), num)  # mean_iou只需要计算union中不为0且不忽略label的
        return mean_iou
 
    return MIOU

if __name__=="__main__":
    a=tf.constant(
        [[0,1,0],
        [0,1,0],
        [1,0,0],
        [0,0,1],
        ],dtype=tf.float32
    )
    b=tf.constant(
        [1,1,0,2]
    )
    print(MIou_score([1,1,1])(b,a))



# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 


def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            