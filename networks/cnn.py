import tensorflow as tf


import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa

class CNN(tf.keras.Model):
    def __init__(self,num_class):
        super().__init__()
        self.numclass=num_class
        #self.norm=tf.keras.layers.LayerNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,             # 卷积层神经元（卷积核）数目
            kernel_size=[7, 7],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.numclass)
    @tf.function()
    def call(self, inputs):
        #x=self.norm(inputs)
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        # x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output