'''
Description: 
Version: 2.0
Autor: CHEN JIE
Date: 2020-10-25 09:52:03
LastEditors: CHEN JIE
LastEditTime: 2020-10-28 10:17:55
language: 
Deep learning framework: 
'''
from matplotlib.pyplot import grid, title
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
1. #****************** 参数设置
# number of samples in the data set
# ratio between training and test sets
# number of iterations of the model
# boundary of the graph
# output directory (the folder must be created on the drive)
N_SAMPLES = 10000
TEST_SIZE = 0.1
N_EPOCHS = 100

GRID_X_START = -1.5
GRID_X_END = 1.5
GRID_Y_START = -1.5
GRID_Y_END = 1.5

OUTPUT_DIR = "binaryclassificationvizualizations"

2.#***************dataset ,we genrate some fake data
X , Y = make_circles(n_samples=N_SAMPLES,noise=.10, factor=.3)
# make_circles()可以在2D中制作一个大圆包含一个小圆。 一个简单的玩具数据集来可视化聚类和分类算法。是sklearn.datasets 模块中的
#factor ：外圈与内圈的尺度因子<1
#noise：表示异常点

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state= 42)
# random_state 相当于随机数种子random.seed() 。random_state 与 random seed 作用是相同的。

#take a look at the dataset of generated dataset
plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
plt.scatter(X_train[:, 0], X_train[:, 1], c =Y_train.ravel(), s=50, cmap= plt.cm.Spectral, edgecolors='black');
plt.savefig('pic/dataset.png')
plt.show()

#Definition of grid boundaries and storage of loss and accuracy history

grid = np.mgrid[GRID_X_START:GRID_X_END:100j, GRID_X_START:GRID_Y_END:100j]
#np.mgrid 功能：返回多维结构，常见的如2D图形，3D图形
#这里即在控制X轴的值，在GRID_X_START = -1.5和GRID_X_END = 1.5之间返回100个值
grid_2d = grid.reshape(2, -1).T
#reshape( )可以塑形，常见到-1，其作用是当不知道某一维有多少元素时，可先用-1指定，最后由其他维自动计算出来。
X, Y = grid
acc_history = []
loss_history = []


#Auxiliary functions,辅助函数

# precede the number with zeros, creating a thong of a certain length
# makeIndexOfLength这个函数是用来设置epoch的序号的和生成每次epoch的可视化图片名
def makeIndexOfLength(index, length):
    indexStr = str(index)
    return ('0' * (length - len(indexStr)) + indexStr)#.length为3，即是位数；第一个epoch，返回001

# the auxiliary function forming graphs of classification boundaries and change of accuracy
def save_model_prediction_graph(epoch, logs):
    prediction_probs = model.predict_proba(grid_2d, batch_size=32, verbose=0)
    plt.figure(figsize=(10,10))
    sns.set_style("whitegrid")
    plt.title('Binary classification with KERAS - epoch: ' + makeIndexOfLength(epoch, 3), fontsize=20)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.contourf(X, Y, prediction_probs.reshape(100, 100), alpha = 0.7, cmap=cm.Spectral)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.ravel(), s=50, cmap=plt.cm.Spectral, edgecolors='black')
    plt.savefig("./" + OUTPUT_DIR + "/keras" + makeIndexOfLength(epoch, 3) + ".png")
    plt.close()


    acc_history.append(logs['accuracy'])
    loss_history.append(logs['loss'])
    plt.figure(figsize=(12,8))
    sns.set_style("whitegrid")
    plt.plot(acc_history)
    plt.plot(loss_history)
    plt.title('Model accuracy and loss - epoch: ' + makeIndexOfLength(epoch, 3), fontsize=20)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim([0,N_EPOCHS])
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.savefig("./" + OUTPUT_DIR + "/loss_acc_" + makeIndexOfLength(epoch, 3) + ".png")
    plt.close()



3.# Creating a KERAS model
model = Sequential()
model.add(Dense(4, input_dim=2,activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Adding callback functions that they will run in every epoch
testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=save_model_prediction_graph)

model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

print(np.shape(X_train))#(900, 2)
print(np.shape(Y_train))#(900,)
print(np.shape(X_train))
print(X_train)
print("*********************************")
print(Y_train)
history = model.fit(X_train, Y_train, epochs=N_EPOCHS, verbose=0, callbacks=[testmodelcb])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

