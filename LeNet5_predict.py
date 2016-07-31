# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:54:05 2016

@author: Akizuki
"""

import time
import chainer
from chainer import Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import scipy as sp
import lenet5
import matplotlib.pyplot as plt

MODEL_PATH = "./LeNet5_Model"
MODEL_NAME = "LeNet5_2.model"

# MNISTデータの読み込み
train, test = chainer.datasets.get_mnist()
train_size = len(train)
test_size = len(test)
label_train = train._datasets[1]
label_test = test._datasets[1]

#データの正規化
data_train = sp.stats.zscore(train._datasets[0], axis=1)
data_test = sp.stats.zscore(test._datasets[0], axis=1)

# データの可視化関数
def draw_image(img):
    plt.imshow(img[0], cmap = plt.get_cmap('gray'), interpolation='none' )
    plt.show()    
    plt.savefig("sample_digtis.png")   
    
# 学習データを画像形式に変換する関数 32x32に拡張もする
def feat2image(feats):
    data = np.ndarray((len(feats), 1, 32, 32), dtype=np.float32)
    
    for i, f in enumerate(feats):
        tmp_data = f.reshape(28, 28) #まず28x28に変換
        min_data = np.min(tmp_data)  #最小値を取得
        data[i,0] = min_data         #最小値で画像を埋める
        data[i,0,2:30,2:30] = tmp_data.copy() #32x32の画像に28x28のデータをコピー
    return data    

#  データの変換（784次元 -> 28x28 -> 32x32）    
data_train = feat2image(data_train)
data_test = feat2image(data_test)

print(label_train[0])
draw_image(data_train[0])

print(label_train[1])
draw_image(data_train[1])



network = lenet5.LeNet5()
model = L.Classifier(network)

# モデルの読み込み
serializers.load_npz(MODEL_PATH+"/"+MODEL_NAME, model)

# 評価
start = time.time()
x_test = Variable(data_test)
result_scores = network(x_test).data
predict_time = time.time() - start
results = np.argmax(result_scores, axis=1)

# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print(predict_time)
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)


