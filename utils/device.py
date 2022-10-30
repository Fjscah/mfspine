import tensorflow as tf

def show_cpu_gpu():
    gpus = tf.config.list_physical_devices(device_type='GPU')
    cpus = tf.config.list_physical_devices(device_type='CPU')
    print("====================CPU=====================\n", cpus)
    print("====================GPU=====================\n", gpus)
    # gpuvliad=tf.test.is_gpu_available()
    # print("GPU whether available :",gpuvliad)
    
    a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
    # b = tf.test.is_gpu_available(
    #     cuda_only=False,
    #     min_cuda_compute_capability=None
    # )  # 判断GPU是否可以用

    print("================CUDA available============\n",a) # 显示True表示CUDA可用
    # print("GPU available:",b) # 显示True表示GPU可用

    # 查看驱动名称
    if tf.test.gpu_device_name():
        print("================Default GPU Device============") # 显示True表示CUDA可用
        print('{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    print("===============================================") # 显示True表示CUDA可用

if __name__=="__main__":
    show_cpu_gpu()