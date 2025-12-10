# dropout_compare_fashion_mnist_team7.py
# Dropout 사용 여부에 따른 과적합/성능 비교 (Team7 버전)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# common, utils 폴더 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.layers import Affine, Relu, SoftmaxWithLoss, Dropout
from common.optimizer import Adam
from utils.mnist_reader import load_mnist


# ====== Team7용 Dropout 지원 네트워크 클래스 ======
class MultiLayerNetDropout:
    """
    - input_size: 입력 차원 (784)
    - hidden_size_list: 예) [100, 100, 100]
    - output_size: 출력 차원 (10)
    - activation: 'relu' or 'sigmoid'
    - weight_init_std: 'he', 'relu', 'xavier', 'sigmoid' 또는 float
    - weight_decay_lambda: L2 weight decay 계수
    - use_dropout: Dropout 사용할지 여부
    - dropout_ratio: Dropout 비율 (예: 0.5)
    """

    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation='relu',
                 weight_init_std='he',
                 weight_decay_lambda=0.0,
                 use_dropout=False,
                 dropout_ratio=0.5):

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
        # activation='relu'만 쓸 거라 사실상 Relu만 사용
        activation_layer = {'sigmoid': Relu, 'relu': Relu}[activation]

        self.layers = OrderedDict()
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list) - 1):
            self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
            self.layers[f"Activation{idx}"] = activation_layer()
            if self.use_dropout:
                self.layers[f"Dropout{idx}"] = Dropout(self.dropout_ratio)

        # 출력층 (Dropout 없음)
        last_idx = self.hidden_layer_num + 1
        self.layers[f"Affine{last_idx}"] = Affine(self.params[f"W{last_idx}"], self.params[f"b{last_idx}"])

        self.last_layer = SoftmaxWithLoss()

    def _init_weight(self, weight_init_std, activation):
        """
        He, Xavier 초기화 지원
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std

            if isinstance(weight_init_std, str):
                if weight_init_std.lower() in ('relu', 'he'):
                    scale = np.sqrt(2.0 / all_size_list[idx - 1])
                elif weight_init_std.lower() in ('sigmoid', 'xavier'):
                    scale = np.sqrt(1.0 / all_size_list[idx - 1])
                else:
                    try:
                        scale = float(weight_init_std)
                    except ValueError:
                        scale = 0.01
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

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

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


# ====== 데이터 로드 함수 ======
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


# ====== 메인 실험 ======
def main():
    # 데이터 로드
    (x_train, t_train), (x_test, t_test), (y_train_int, y_test_int) = load_fashion_mnist(
        normalize=True,
        one_hot_label=True
    )

    # train 일부만 사용 (속도)
    train_size = 10000
    x_train = x_train[:train_size]
    t_train = t_train[:train_size]
    y_train_int = y_train_int[:train_size]

    # train / val 나누기 (과적합 확인용)
    val_size = 2000
    x_val = x_train[-val_size:]
    t_val = t_train[-val_size:]
    x_train_use = x_train[:-val_size]
    t_train_use = t_train[:-val_size]
    y_train_use_int = y_train_int[:-val_size]

    print(f"train: {x_train_use.shape[0]}, val: {x_val.shape[0]}, test: {x_test.shape[0]}")

    # 실험 설정: Dropout 사용 / 미사용
    configs = {
        "no_dropout": {
            "use_dropout": False,
            "dropout_ratio": 0.0,
        },
        "dropout0.5": {
            "use_dropout": True,
            "dropout_ratio": 0.5,
        },
    }

    networks = {}
    optimizers = {}
    train_acc_history = {k: [] for k in configs.keys()}
    val_acc_history = {k: [] for k in configs.keys()}

    # weight decay는 이전에 찾은 best 값 사용
    weight_decay_lambda = 1e-5

    for name, cfg in configs.items():
        print(f"{name} : use_dropout={cfg['use_dropout']}, ratio={cfg['dropout_ratio']}")
        networks[name] = MultiLayerNetDropout(
            input_size=784,
            hidden_size_list=[100, 100, 100],  # 3층 MLP
            output_size=10,
            activation='relu',
            weight_init_std='he',
            weight_decay_lambda=weight_decay_lambda,
            use_dropout=cfg["use_dropout"],
            dropout_ratio=cfg["dropout_ratio"],
        )
        optimizers[name] = Adam(lr=0.001)

    # 학습 설정
    max_epochs = 20
    batch_size = 128
    iter_per_epoch = max(x_train_use.shape[0] // batch_size, 1)

    print("=== Dropout 비교 실험 시작 ===")
    for epoch in range(1, max_epochs + 1):
        for _ in range(iter_per_epoch):
            batch_mask = np.random.choice(x_train_use.shape[0], batch_size)
            x_batch = x_train_use[batch_mask]
            t_batch = t_train_use[batch_mask]

            for name in configs.keys():
                network = networks[name]
                optimizer = optimizers[name]

                grads = network.gradient(x_batch, t_batch)
                optimizer.update(network.params, grads)

        # epoch마다 train / val accuracy 측정
        for name in configs.keys():
            network = networks[name]
            train_acc = network.accuracy(x_train_use, y_train_use_int)
            val_acc = network.accuracy(x_val, np.argmax(t_val, axis=1))
            train_acc_history[name].append(train_acc)
            val_acc_history[name].append(val_acc)

        print(f"[Epoch {epoch:02d}]")
        for name in configs.keys():
            print(f" {name:10s} | train_acc={train_acc_history[name][-1]:.4f}, "
                  f"val_acc={val_acc_history[name][-1]:.4f}")

    # 최종 test accuracy
    print("\n=== 최종 Test Accuracy (Dropout 비교) ===")
    for name in configs.keys():
        acc = networks[name].accuracy(x_test, y_test_int)
        print(f"[{name}] Test Accuracy: {acc:.4f}")

    # Accuracy 그래프 그리기
    epochs = np.arange(1, max_epochs + 1)
    plt.figure(figsize=(8, 5))
    for name in configs.keys():
        plt.plot(epochs, train_acc_history[name], '--', label=f"{name}_train")
        plt.plot(epochs, val_acc_history[name], label=f"{name}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Effect of Dropout on Train / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dropout_compare_accuracy.png")
    plt.show()
    print("Dropout 비교 그래프가 'dropout_compare_accuracy.png' 로 저장되었습니다.")


if __name__ == "__main__":
    main()
