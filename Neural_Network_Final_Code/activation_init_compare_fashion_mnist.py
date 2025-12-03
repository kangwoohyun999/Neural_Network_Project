# activation_init_compare_fashion_mnist.py
# 활성화 함수와 가중치 초기화 방법 조합 비교 실험

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ====== 경로 설정 (common, utils 불러오기) ======
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam
from utils.mnist_reader import load_mnist


# ====== One-hot 변환 함수 ======
def to_one_hot(y, num_classes=10):
    t = np.zeros((y.size, num_classes), dtype=np.float32)
    t[np.arange(y.size), y] = 1.0
    return t


# ====== Fashion-MNIST 로더 ======
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


def main():
    # 1. 데이터 로드
    (x_train, t_train), (x_test, t_test), (y_train_int, y_test_int) = load_fashion_mnist(
        normalize=True,
        one_hot_label=True,
    )

    # 속도 위해 train 일부만 사용 (예: 10,000개)
    train_size = 10000
    x_train = x_train[:train_size]
    t_train = t_train[:train_size]
    y_train_int = y_train_int[:train_size]

    print(f"사용하는 train 데이터 개수: {train_size}")

    # 2. 비교할 조합 설정
    #  - activation: 'sigmoid' 또는 'relu'
    #  - weight_init_std:
    #       'xavier' 또는 'sigmoid'  → Xavier 초기화
    #       'he' 또는 'relu'         → He 초기화
    combo_settings = {
        "sigmoid+xavier": {
            "activation": "sigmoid",
            "weight_init_std": "xavier",
        },
        "sigmoid+he": {
            "activation": "sigmoid",
            "weight_init_std": "he",
        },
        "relu+xavier": {
            "activation": "relu",
            "weight_init_std": "xavier",
        },
        "relu+he": {
            "activation": "relu",
            "weight_init_std": "he",
        },
    }

    networks = {}
    optimizers = {}
    loss_history = {}
    test_acc_result = {}

    # weight decay는 여기서는 0으로 두고,
    # activation / 초기화의 영향만 보는 실험으로 설정
    weight_decay_lambda = 0.0

    for key, cfg in combo_settings.items():
        print(f"{key} : act={cfg['activation']}, init={cfg['weight_init_std']}")
        networks[key] = MultiLayerNet(
            input_size=784,
            hidden_size_list=[100, 100, 100],   # 3-layer 예시
            output_size=10,
            activation=cfg["activation"],
            weight_init_std=cfg["weight_init_std"],
            weight_decay_lambda=weight_decay_lambda,
        )
        optimizers[key] = Adam(lr=0.001)
        loss_history[key] = []

    # 3. 학습 설정
    max_iterations = 2000
    batch_size = 128

    print("=== Activation + Initialization 조합 비교 실험 시작 ===")
    print(f"max_iterations = {max_iterations}, batch_size = {batch_size}")

    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in combo_settings.keys():
            network = networks[key]
            optimizer = optimizers[key]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            loss = network.loss(x_batch, t_batch)
            loss_history[key].append(loss)

        # 200 iteration마다 로그 출력
        if i % 200 == 0:
            print(f"========== iteration:{i} ==========")
            for key in combo_settings.keys():
                network = networks[key]
                current_loss = loss_history[key][-1]
                train_acc = network.accuracy(x_train, y_train_int)
                print(f"{key} | loss: {current_loss:.4f}, train_acc: {train_acc:.4f}")

    # 4. 최종 Test Accuracy
    print("\n=== 최종 Test Accuracy (Activation + Init 비교) ===")
    for key in combo_settings.keys():
        network = networks[key]
        acc = network.accuracy(x_test, y_test_int)
        test_acc_result[key] = acc
        print(f"[{key}] Test Accuracy: {acc:.4f}")

    # 5. Loss 곡선 그래프
    plt.figure(figsize=(8, 5))
    iters = np.arange(max_iterations)
    for key in combo_settings.keys():
        plt.plot(iters, loss_history[key], label=key)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Effect of Activation & Initialization (Train Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("activation_init_compare_loss.png")
    print("Loss 그래프가 'activation_init_compare_loss.png' 이름으로 저장되었습니다.")

    # 6. Test Accuracy 막대 그래프
    plt.figure(figsize=(7, 4))
    keys = list(test_acc_result.keys())
    values = [test_acc_result[k] for k in keys]
    plt.bar(keys, values)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Activation + Initialization")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Activation & Initialization")
    plt.tight_layout()
    plt.savefig("activation_init_compare_accuracy.png")
    print("정확도 그래프가 'activation_init_compare_accuracy.png' 이름으로 저장되었습니다.")
    
    plt.show()


if __name__ == "__main__":
    main()
