# -*- coding: utf-8 -*-

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

#フォルダ作成
import os
try:
    os.mkdir(MODEL_PATH)
except:
    pass

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

# 学習のさせかたの指定
optimizer = optimizers.SGD()
optimizer.setup(model)

# 学習
n_epoch = 100  # 学習繰り返し回数
batchsize = 20  # 学習データの分割サイズ
N_train = len(data_train)
N_test = len(data_test)
losses = []  # 各回での誤差の変化を記録するための配列

start = time.time()  # 処理時間の計測開始
for epoch in range(n_epoch):
    print('epoch: %d' % (epoch+1))
    perm = np.random.permutation(N_train)  # 分割をランダムにするための並べ替え
    #print(perm) #選択した学習データのIDを表示
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_train, batchsize):
        # 並べ替えた i〜i+batchsize 番目までのデータを使って学習
        x_batch = data_train[perm[i:i+batchsize]]
        t_batch = label_train[perm[i:i+batchsize]]

        # 勾配の初期化
        optimizer.zero_grads()

        # 順伝播
        x = Variable(x_batch)
        t = Variable(t_batch)
        loss = model(x,t)
        
        # 誤差逆伝播
        loss.backward()

        # 認識率の計算（表示用）
        accuracy = model.accuracy

        # パラメータ更新
        optimizer.update()

        sum_loss += float(loss.data) * batchsize
        sum_accuracy += float(accuracy.data) * batchsize

    losses.append(sum_loss / N_train)
    print("loss: %f, accuracy: %f" % (sum_loss / N_train, sum_accuracy / N_train))
    ##モデルの保存
    serializers.save_npz(MODEL_PATH+"/LeNet5_{0}.model".format(epoch), model)

training_time = time.time() - start


# 評価
start = time.time()
x_test = Variable(data_test)
result_scores = network(x_test).data
predict_time = time.time() - start
results = np.argmax(result_scores, axis=1)

# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print(training_time, predict_time)
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)


