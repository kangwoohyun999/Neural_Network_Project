# train_fashion_mnist_team7_final.py
# Team7 최종 학습 코드 (Dropout + Weight Decay + LR Scheduler)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# ====== 경로 설정 ======
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.layers import Affine, Relu, SoftmaxWithLoss, Dropout, Sigmoid
from common.optimizer import Adam
from utils.mnist_reader import load_mnist


# ====== One-hot & 데이터 로드 ======
def to_one_hot(y, num_classes=10):
    t = np.zeros((y.size, num_classes), dtype=np.float32)
    t[np.arange(y.size), y] = 1.0
    return t


def load_fashion_mnist(normalize=True, one_hot_label=True):
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'fashion')
    X_train, y_train = load_mnist(data_dir, kind='train')
    X_test, y_test = load_mnist(data_dir, kind='t10k')

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if one_hot_label:
        t_train = to_one_hot(y_train, num_classes=10)
        t_test = to_one_hot(y_test, num_classes=10)
    else:
        t_train, t_test = y_train, y_test

    return (X_train, t_train), (X_test, t_test), (y_train, y_test)


# ====== Dropout 지원 MultiLayerNet ======
class MultiLayerNetDropout:
    """
    Team7용 최종 네트워크
    - Dropout, Weight Decay, He/Xavier 초기화 지원
    """

    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation='relu',
                 weight_init_std='he',
                 weight_decay_lambda=0.0,
                 use_dropout=False,
                 dropout_ratio=0.3):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.params = {}

        # 가중치 초기화
        self._init_weight(weight_init_std, activation)

        # 계층 생성
        act_layer = {'sigmoid': Sigmoid, 'relu': Relu}[activation]

        self.layers = OrderedDict()
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list) - 1):
            self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
            self.layers[f"Activation{idx}"] = act_layer()
            if self.use_dropout:
                self.layers[f"Dropout{idx}"] = Dropout(self.dropout_ratio)

        # 출력층 (Dropout 없음)
        last_idx = self.hidden_layer_num + 1
        self.layers[f"Affine{last_idx}"] = Affine(self.params[f"W{last_idx}"], self.params[f"b{last_idx}"])

        self.last_layer = SoftmaxWithLoss()

    def _init_weight(self, weight_init_std, activation):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std

            if isinstance(weight_init_std, str):
                name = weight_init_std.lower()
                if name in ('relu', 'he'):
                    scale = np.sqrt(2.0 / all_size_list[idx - 1])
                elif name in ('sigmoid', 'xavier'):
                    scale = np.sqrt(1.0 / all_size_list[idx - 1])
                else:
                    scale = 0.01
            else:
                scale = float(weight_init_std)

            self.params[f"W{idx}"] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params[f"b{idx}"] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg=train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg=train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params[f"W{idx}"]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f"W{idx}"] = self.layers[f"Affine{idx}"].dW + self.weight_decay_lambda * self.params[f"W{idx}"]
            grads[f"b{idx}"] = self.layers[f"Affine{idx}"].db

        return grads


# ====== 메인 학습 루프 ======
def main():
    # 1. 데이터 로드
    (x_train, t_train), (x_test, t_test), (y_train_int, y_test_int) = load_fashion_mnist(
        normalize=True,
        one_hot_label=True,
    )

    # train 전체 사용 (60,000), 그 중 일부를 val로 분리
    train_size = x_train.shape[0]        # 60000
    val_size = 10000                     # 검증 1만 개
    x_val = x_train[-val_size:]
    t_val = t_train[-val_size:]
    x_train_use = x_train[:-val_size]
    t_train_use = t_train[:-val_size]
    y_train_use_int = y_train_int[:-val_size]

    print(f"train: {x_train_use.shape[0]}, val: {x_val.shape[0]}, test: {x_test.shape[0]}")

    # 2. 네트워크 & Optimizer 설정
    network = MultiLayerNetDropout(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],   # 4층 MLP
        output_size=10,
        activation='relu',
        weight_init_std='he',
        weight_decay_lambda=1e-5,
        use_dropout=True,
        dropout_ratio=0.3,
    )

    optimizer = Adam(lr=0.001)

    # 3. 학습 하이퍼파라미터
    max_epochs = 25
    batch_size = 128
    iter_per_epoch = max(x_train_use.shape[0] // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    print("=== Team7 최종 학습 시작 ===")
    print(f"epochs={max_epochs}, batch_size={batch_size}, "
          f"dropout=0.3, weight_decay=1e-5, optimizer=Adam")

    for epoch in range(1, max_epochs + 1):
        # ---- (1) Learning Rate Scheduler ----
        if epoch == 15 or epoch == 20:
            optimizer.lr *= 0.5
            print(f"[Epoch {epoch}] learning rate decayed -> {optimizer.lr}")

        # ---- (2) 미니배치 학습 ----
        for _ in range(iter_per_epoch):
            batch_mask = np.random.choice(x_train_use.shape[0], batch_size)
            x_batch = x_train_use[batch_mask]
            t_batch = t_train_use[batch_mask]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

        # ---- (3) Epoch마다 정확도 측정 ----
        train_acc = network.accuracy(x_train_use, y_train_use_int)
        val_acc = network.accuracy(x_val, np.argmax(t_val, axis=1))
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"[Epoch {epoch:02d}] loss={loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # 4. 최종 Test Accuracy
    test_acc = network.accuracy(x_test, y_test_int)
    print("\n=== 최종 Test Accuracy (Team7 최종 모델) ===")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 5. 그래프 저장
    # (1) Loss vs Iteration
    plt.figure(figsize=(8, 5))
    iters = np.arange(len(train_loss_list))
    plt.plot(iters, train_loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Team7 Final Model - Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("team7_final_loss.png")

    # (2) Train / Val Accuracy vs Epoch
    epochs = np.arange(1, max_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc_list, label="train_acc")
    plt.plot(epochs, val_acc_list, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Team7 Final Model - Train / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("team7_final_accuracy.png")

    plt.show()
    print("그래프가 'team7_final_loss.png', 'team7_final_accuracy.png' 으로 저장되었습니다.")


if __name__ == "__main__":
    main()
