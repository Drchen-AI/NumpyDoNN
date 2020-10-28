'''
Description: 
Version: 2.0
Autor: CHEN JIE
Date: 2020-10-27 14:45:34
LastEditors: CHEN JIE
LastEditTime: 2020-10-27 20:33:37
language: 
Deep learning framework: 
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Settings
# learning rate
LR = 0.04
# parameters a and b of the real function
REAL_PARAMS = [1.2, 2.5]
# starting point for gradient descent
INIT_PARAMS = [-1, -1.5]
# output directory (the folder must be created on the drive)
OUTPUT_DIR = "gradient_descent"


# 1.辅助函数
# precede the number with zeros, creating a thong of a certain length
def makeIndexOfLength(index, length):
    indexStr = str(index)
    return ('0' * (length - len(indexStr)) + indexStr)


#2.Performing the simulation
x = np.linspace(-1, 1, 200, dtype=np.float32)

y_fun = lambda a, b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))

noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise

# tensorflow graph
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a, b)
mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(180):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_); b_list.append(b_); cost_list.append(mse_)
        result, _ = sess.run([pred, train_op])


# Creates visualization
# 3D cost figure
for angle in range(0, 180):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    a3D, b3D = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))  # parameter space
    cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
    ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.6)
    ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
    ax.set_xlabel('a'); ax.set_ylabel('b')
    ax.plot(a_list[:angle], b_list[:angle], zs=cost_list[:angle], zdir='z', c='r', lw=3)    # plot 3D gradient descent
    ax.view_init(30 + (90 - angle)/5, 45 + angle*2)
    plt.savefig("./" + OUTPUT_DIR + "/" + makeIndexOfLength(angle, 3) + ".png")
    plt.close()
    print("执行成功",angle)