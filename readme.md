# directory structure

+---config  
|-parameter profile  
+---dataset  
|-dataloader: loader data and split data into train/valid/test folder  
|-enhancer:augmentation  
|-localthreshold  
+---models : store trained model  
|-store models  
+---networks  
|-some networks  
+---trainers  
|-loss  
|-metrics  
|-trainer  
|-test  
+---transformer  
| no use  
+---utils  
|-yaml_config : load default yaml profile, you need change it to yours. some yaml profile templates in config folder  
|-file_base  

# usage
0. anotation 
   
   you need install [spine-segment-pipeline pluin](/spine-segment-pipeline)

1. set yaml config profile
   
    **change utils.yaml_config.py `default_configuration` path**

   You must configure it first, otherwise none of the subsequent steps will run. Some ref yaml profile templates in config folder 
  

2. image augmentation
   
   **run python dataset/enhander.py** 

   will crop ori image and label from `oridata_path` and `orilabel_path` into size of (`input_sizez`,`input_sizexy`,`input_sizexy`) and store in `data_path` and `label_path` respectively. when stroed in lebel_path, filename will end with `label_suffix`.
   
   if use `unet` Trainning model , interger instance label will all set 2, dendrite set 1, background set 0.

   if use other : todo

   would get (iter *number of oriimgs) new crop imgs. so you need set iter according to your dataset size 

3. split dataset into train and test
   
   **run python dataset/dataloader.py**

    script will split data into three part: train , valid, test accoding to user specified `partion` and store in  `Train_path` folder for trainning.

   here only use most simple method -- split img to different folder. ( I want to make tfrecord to increase img load velocity , but tfrecord cannot be read correctly )

4. train model 
   
   **run python trainers/trainer.py**

   if you have pretrained mode, change `premodel` variable in trainer.py to your model path

   LOG :
   1. loss_time folder     : loss value with epoch 
   2. model.h5             : weight model
   3. all.log              : some infomation log 
   4. error log            : excution error log
   5. events.out.tfevents  : for tensorboard
      ```cmd
      tensorboard --logdir=events.out.tfevents folder
      ```


5. test predict
   
   **run python trainers/test.py** 

   will predict img in test folder, and show img and results.
   You could check model in this way


6. patch process time imgs(jupyter notebook)
   
   see notebook  in "spine-ipynb" folder for details
