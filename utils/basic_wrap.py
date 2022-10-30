#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_wrap.py
@Time    :   2022/04/19 22:53:28
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   GNU, LNP2 group
@Desc    :   some wrap for time log and so on
'''
# here put the import lib
# descrbe.py
import functools
import logging
import time
from .yaml_config import default_configuration,YAMLConfig



def timing(func):
    """this is outer clock function"""
    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function"""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        print("FUNC NAME : "+func.__name__ + " RUN TIME COST -> {}s".format(time_cost))
        return result
    return clocked  # --> 3

 
def create_logger(filename="excute.log"):
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("excute_logger")
    logger.setLevel(logging.INFO)
 
    # create the logging file handler
    fh = logging.FileHandler(filename)
 
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
 
    # add handler to logger object
    logger.addHandler(fh)
    return logger
 
def logit(file="excute.log"):
    def exception(function):
        """
        A decorator that wraps the passed in function and logs 
        exceptions should one occur
        """
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            logger = create_logger(filename=file)
            try:
                return function(*args, **kwargs)
            except Exception as e:
                # log the exception
                err = "There was an exception in  "
                err += function.__name__
                logger.exception(err)
                #logger.exception(e)
    
                # re-raise the exception
                raise
        return wrapper
    return exception

from logging import handlers
import os





class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射
    def setting(self, configuration: YAMLConfig = None):
    
        if configuration == None:
            configuration = default_configuration
        self.configuration = configuration
        self.log_path=self.configuration.config["Path"]["log_path"]
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        # self.setting(configuration)
        self.filename=filename#os.path.join(os.path.abspath(self.log_path),"all.log")
        self.logger = logging.getLogger(self.filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=self.filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)#设置文件里写入的格式
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
# def get_log(path):
#     formatter = logging.Formatter(fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.INFO)
#     logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
#                     filename=path,
#                     filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     #a是追加模式，默认如果不写的话，就是追加模式
#                     format=
#                     '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                     #日志格式
#                     )
#     handler = logging.StreamHandler()
#     handler.setFormatter(formatter)
#     logger = logging.getLogger('info')
#     logger.addHandler(handler)
#     return logger

# TensorBoard：训练过程可视化




#/////////////////////////////////////////////////////////////////////
@functools.lru_cache()  # --> 5
@timing  # --> 6
def fib(n):
    """this is fibonacci function"""
    return n if n < 2 else fib(n - 1) + fib(n - 2)

@logit("test_degub.log")
def zero_divide():
    1 / 0

if __name__ == "__main__":
    # 如果有 @functools.wraps(func)  # --> 4，大多数情况下我们希望的输出是这样的
    fib(1) # 输出 fib func time_cost -> 9.5367431640625e-07
    print(fib.__name__)  # 输出 fib
    print(fib.__doc__)  # 输出 this is fibonacci function

    # # 如果没有@functools.wraps(func)  # --> 4
    # fib(1) # 输出 fib func time_cost -> 9.5367431640625e-07
    # print(fib.__name__)  # 输出 clocked
    # print(fib.__doc__)  # 输出 this is inner clocked function
    
    log = Logger('all.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error.log', level='error').logger.error('error')
    
    zero_divide()