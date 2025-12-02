# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.util import shuffle_dataset
from common.trainer import Trainer

# 0. MNIST 데이터 읽기==========
from data import mnist_reader
x_train, t_train = mnist_reader.load_mnist('../data/fashion/', kind='train')
x_test, t_test = mnist_reader.load_mnist('../data/fashion/', kind='t10k')

# 10%를 검증 데이터로 분할
validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, dropout, epocs=20):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100],
                            output_size=10, weight_decay_lambda=weight_decay, activation='relu', weight_init_std='relu',
                 use_dropout = True, dropout_ration = dropout, use_batchnorm=True)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='adam', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 50
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 탐색한 하이퍼파라미터의 범위 지정===============
    weight_decay = 10**np.random.uniform(-8,-3)
    lr = 10**np.random.uniform(-4, -2)
    dropout = np.random.uniform(0.2,0.5)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay, dropout)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay)+", dropout_ration:" + str(dropout))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay) + ", dropout_ration:" + str(dropout)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

plt.figure(figsize=(10, 8))

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig('hyperparam_optim.png')  
plt.show()
