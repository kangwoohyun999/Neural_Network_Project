import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
# 0. MNIST 데이터 읽기==========
from data import mnist_reader
x_train, t_train = mnist_reader.load_mnist('../data/fashion/', kind='train')
x_test, t_test = mnist_reader.load_mnist('../data/fashion/', kind='t10k')
# 확인을 빠르게 하기위해 학습 데이터 수를 줄임
x_train = x_train[:10000]
t_train = t_train[:10000]

# 드롭아웃 사용 유무와 비율, 가중치 감소 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.3
weight_decay_lambda = 0.001 # 가중치 감쇠 설정
# ====================================================

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio,weight_decay_lambda=weight_decay_lambda, use_batchnorm=True)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=201, mini_batch_size=100,
                  optimizer='adam', optimizer_param={'lr': 0.001}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig("weight_decay_and_dropout.png")
plt.show()