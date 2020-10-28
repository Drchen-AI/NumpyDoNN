'''
Description: 
Version: 2.0
Autor: CHEN JIE
Date: 2020-10-27 17:12:25
LastEditors: CHEN JIE
LastEditTime: 2020-10-27 17:12:28
language: 
Deep learning framework: 
'''
import tensorflow as tf
import numpy as np
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


print(tf.__version__)
gpus =tf.config.experimental.list_physical_devices(device_type= 'GPU')
cpus= tf.config. experimental.list_physical_devices(device_type= 'CPU')
print(gpus)
print(cpus)
print("tensorflow-gpu:",tf.test.is_gpu_available())

#指定在GP上执行随机数操作
with tf.device('/gpu:0'):
    gpu_a= tf.random.normal([10000, 1000])
    gpu_b= tf.random.normal([1000,2000])
    gpu_c = tf.matmul(gpu_a, gpu_b)

print("gpu_a:",gpu_a.device)
print("gpu b:", gpu_b.device)
print("gpu c:", gpu_c.device)
