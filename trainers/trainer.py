import gc
import logging
import os
import sys
import warnings
from venv import create

# import cv2
# import cv2
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import io
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

matplotlib.use('Agg')
import numpy as np
import scipy.signal
from tensorflow import keras
from tensorflow.keras import backend as K

# tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, RMSprop
from tensorflow.python.keras import Model
from tensorflow_addons.optimizers import AdamW
from tqdm import tqdm
from xgboost import train

sys.path.append(".")
from trainers.metrics import  MIou_score, dice_coef,Iou_score,MeanIoU,cal_mean_iou
from keras.metrics import categorical_accuracy
from dataset.dataloader import CusDataset, SliceLoader, load_datate_from_folder, DataLoader, UnetDataset
from dataset.tfrecord import TFRecordDataset
from networks import unet
from utils.file_base import create_dir
from utils.basic_wrap import Logger,logit
from utils.yaml_config import YAMLConfig, default_configuration
from trainers import loss
from utils.basic_wrap import timing
from trainers.loss import *
gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = True

optimizer_dict = {
    'Adam': Adam,
    'AdamW': AdamW,
    'Adagrad': Adagrad,
    'RMSProp': RMSprop,
    'SGD': SGD,
}

loss_dict={
    
}


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir, val_loss_flag=True):
        import datetime
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(),
                                                   '%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, "loss_" + str(self.time_str))
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        with open(
                os.path.join(self.save_path,
                             "epoch_loss_" + str(self.time_str) + ".txt"),
                'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")

        if self.val_loss_flag:
            self.val_loss.append(logs.get('val_loss'))
            with open(
                    os.path.join(
                        self.save_path,
                        "epoch_val_loss_" + str(self.time_str) + ".txt"),
                    'a') as f:
                f.write(str(logs.get('val_loss')))
                f.write("\n")

        self.loss_plot()
        if (logs.get('accuracy') > 0.96):
            print("\n accuracy up to  96% ！")
            self.model.stop_training = True

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()

        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        try:
            plt.plot(
                iters,
                scipy.signal.savgol_filter(
                    self.losses, 5 if len(self.losses) < 25 else 15, 3),
                'green',
                linestyle='--',
                linewidth=2,
                label='smooth train loss')
        except:
            pass

        if self.val_loss_flag:
            plt.plot(
                iters, self.val_loss, 'coral', linewidth=2, label='val loss')
            try:
                plt.plot(
                    iters,
                    scipy.signal.savgol_filter(
                        self.val_loss, 5 if len(self.losses) < 25 else 15, 3),
                    '#8B4513',
                    linestyle='--',
                    linewidth=2,
                    label='smooth val loss')
            except:
                pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(
            os.path.join(self.save_path,
                         "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")





class Trainer:
    EPOCH_PHASE = 0
    ITERATION_PHASE = 1

    def setting(self, configuration: YAMLConfig = None):

        if configuration == None:
            configuration = default_configuration
        self.configuration = configuration
        dict_a = self.configuration.config["Path"]
        dict_b = self.configuration.config["Training"]
        dict_c = self.configuration.config["Data"]
        #print(dict_b)
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.Train_path=os.path.abspath(self.Train_path)
        if "z" in self.axes:
            self.imgshape = (self.input_sizez, self.input_sizexy,
                             self.input_sizexy,1)
        else:
            self.imgshape = (self.input_sizexy, self.input_sizexy,1)
        self.logger=Logger(os.path.join(os.path.abspath(self.log_path),"model_info.log"))
        keys = ["model", "batch_size", "cls_weight","label_suffix", 
                "learning_rate","optimizer_name","loss_type","imgshape"]
        kvs = [k + "\t:\t" + str(getattr(self, k)) for k in keys]
        self.logger.logger.info(
            "\n=========YAML TRAIN INFO==============\n"+
            "\n".join(kvs)+
            "\n======================================\n"
        )
    
        self.inital_model()
        self.initial_optimizer()
        self.initial_metric()
        self.initial_loss()

    def initial_loss(self):
        self.cls_weight
        #   建议选项：
        #   种类少（几类）时，设置为True
        #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
        #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
        #------------------------------------------------------------------#
        dice_loss = False
        #------------------------------------------------------------------#
        #   是否使用focal loss来防止正负样本不平衡
        #------------------------------------------------------------------#
        ffocal_loss = False
        self.loss = weighted_categorical_crossentropy(self.cls_weight)

    def initial_optimizer(self):
        self.optimizer = optimizer_dict[self.optimizer_name](
            learning_rate=self.learning_rate)

    def inital_model(self):
        network_type = self.configuration.get_entry(['Training', 'model'])
        if "unet3d" == network_type:
            self.model = unet.UNet3D(self.configuration)
        elif "unet2d" == network_type:
            self.model = unet.UNet2D(self.configuration)
        elif "sunet_2D"==network_type:
            self.model=unet.sunet_2D(self.imgshape)

    def initial_metric(self):
        pass

    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)
        #self.logger = logging.getLogger('info')
        # self.model.compile(optimizer=self.optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        #self.log = log
        self.num_train_examples = 0
        self.best_val_f1_scores = [0, 0, 0]
        self.best_val_epoch = 0
    
    def get_summary(self):
        def _s():
            s=""
            def getsummary(string,get=False):
                nonlocal s
                s+="\n"+string
                if get: return s
            return getsummary
        _f=_s()
        self.model.summary(print_fn=_f)
        self.logger.logger.info(_f("",1))
        
    
    @logit("error.log")
    def train(self, premodel=""):
        #trdatasets=CusDataset(self.configuration)
        #trdatasets,vadatasets,tedatasets=tfd.load_train_test(num_class=self.num_classes)
        data_loader = SliceLoader()
        
        trdatasets = UnetDataset(self.Train_path + "\\train", self.imgshape,self.label_suffix,
                                 self.batch_size, self.num_classes)
        vadatasets = UnetDataset(self.Train_path + "\\valid", self.imgshape,self.label_suffix,
                                 self.batch_size, self.num_classes)

        #trdatasets,vadatasets,tedatasets=data_loader.get_dataset()

        # @timing
        # def todataset(ds):
        #     return tf.data.Dataset.from_tensor_slices(ds)

        # trdatasets=todataset(trdatasets)
        # vadatasets=todataset(vadatasets)
        # trdatasets,vadatasets,tedatasets=load_datate_from_folder(default_configuration)
        #trdatasets=trdatasets.shuffle(buffer_size=100).batch(self.batch_size)
        #trdatasets=trdatasets.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # checkpoint = tf.train.Checkpoint(myModel=self.model,myOptimizer=self.optimizer,
        #                                  save_best_only=False, save_weights_only=False,save_frequency=1)
        #manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3,)
        #checkpoint.restore(tf.train.latest_checkpoint('./save'))
        #checkpoint.restore(save_path_with_prefix_and_index)
        #print(trdatasets)

        #-----------------------#
        #   Log  #
        #-----------------------#
        history = LossHistory(self.log_path)
        create_dir(self.model_path)
        ccheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path + 'ep{epoch:03d}-loss{loss:.3f}.h5',
            save_weights_only=True,
            max_to_keep=5,
            monitor='val_loss',
            verbose=2,
            save_best_only=False,
            period=1)
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_path, write_graph=True)
        #-----------------------#
        #   Loss  #
        #-----------------------#
        #print("===========",self.num_classes)
        cls_weights = np.array([1, 1, 1], np.float32)
        loss = CE(cls_weights)
        num_workers = 4
        
        
        print(tf.executing_eagerly())

        if "z" in self.axes:  # run 3d trainning
            # self.model.compile(optimizer=self.optimizer,
            #                    loss="sparse_categorical_crossentropy",#loss,#tf.keras.losses.sparse_categorical_crossentropy,#loss.dice_loss_with_CE([1,2,1]),
            #                    metrics=['accuracy',"mean_iou"])

            self.model.build(
                input_shape=(self.batch_size, self.input_sizez,
                             self.input_sizexy, self.input_sizexy, 1))

            self.model.run_eagerly = True
            # self.model.fit(trdatasets,
            #                         epochs=50, verbose=2,
            #                         callbacks=[history,ccheckpoint],
            #                         batch_size=self.batch_size)

            self.model.fit(
                x=trdatasets[0],
                y=trdatasets[1],
                epochs=self.epochs,
                verbose=2,
                # validation_data=vadatasets,
                #use_multiprocessing = True if num_workers > 1 else False,
                validation_data=vadatasets,
                steps_per_epoch=50,
                callbacks=[history, ccheckpoint, tensorboard_callback],
            )
        else:  # run 2d trainning
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,#"categorical_crossentropy",#"categorical_crossentropy(too slow unbalance not reasonal)",#tf.keras.losses.sparse_categorical_crossentropy,#"sparse_categorical_crossentropy",#loss,#tf.keras.losses.sparse_categorical_crossentropy,#loss.dice_loss_with_CE([1,2,1]),
                metrics=[
                    'accuracy', #"categorical_accuracy",
                    #MeanIoU(num_classes=3),
                    cal_mean_iou(num_classes=3, ignore_labels=[0]),
                    Iou_score(),
                    MIou_score(cls_weights, smooth=1e-5, threhold=0.5)
                ])
            # writer = tf.summary.create_file_writer("logdir")
            # tf.summary.trace_on(graph=True, profiler=True)

            self.model.run_eagerly = True
            self.model.build(
                input_shape=(self.batch_size, self.input_sizexy,
                             self.input_sizexy, 1))
            self.get_summary()
            

            # graph_writer = tf.summary.create_file_writer(logdir="logdir")
            # from tensorflow.python.ops import summary_ops_v2  # need import thid module
            # with graph_writer.as_default():
            #     graph = self.model.call.get_concrete_function(
            #         trdatasets[0]).graph
            #     summary_ops_v2.graph(graph.as_graph_def())
            # graph_writer.close()

            #-----------------------#
            #   Load model weights  #
            #-----------------------#
            checkpoint_save_path = premodel
            #checkpoint_save_path = ""  #r"models\M2d_seg\modelep100-loss0.011.h5"  # 模型参数保存路径
            if checkpoint_save_path and os.path.exists(checkpoint_save_path):
                self.model.load_weights(checkpoint_save_path)
                self.logger.logger.info(
                    "\n==============LOAD PRETRAINED model==============\n"+
                    checkpoint_save_path+
                    "\n=================================================\n"
                )
               

            #-----------------------#
            #   Fit  #
            #-----------------------#
            #self.model.fit(trdatasets[0], epochs=100)
            self.model.fit_generator(
                generator=trdatasets,
                steps_per_epoch=30,
                epochs=self.epochs,
                verbose=1,
                #use_multiprocessing = True ,
                validation_data=vadatasets,
                callbacks=[history, ccheckpoint, tensorboard_callback],
            )
        
        #usually no use
        tf.keras.models.save_model(self.model, self.model_path)
        history.loss_plot()
        # checkpoint.save(self.model_save_path)


if __name__ == "__main__":
    trainmodel = Trainer()

    premodel =r"D:\code\myspine\models\MM2d_seg\modelep139-loss0.094.h5"
    trainmodel.train(premodel)
