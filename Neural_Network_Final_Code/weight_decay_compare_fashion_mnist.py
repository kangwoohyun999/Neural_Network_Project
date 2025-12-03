# weight_decay_compare_fashion_mnist.py
# Adam + 서로 다른 weight_decay_lambda 비교 실험

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ====== 경로 설정 ======
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam
from utils.mnist_reader import load_mnist


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


def main():
    # 1. 데이터 로드
    (x_train, t_train), (x_test, t_test), (y_train_int, y_test_int) = load_fashion_mnist(
        normalize=True,
        one_hot_label=True,
    )

    # 일부만 사용해서 속도 줄이기
    train_size = 10000
    x_train = x_train[:train_size]
    t_train = t_train[:train_size]

    print(f"사용하는 train 데이터 개수: {train_size}")

    # 2. 비교할 weight decay 값들
    #   0 : 정규화 없음
    #   그 외 : L2 정규화 강도
    weight_decays = [0.0, 1e-5, 1e-4, 1e-3]

    networks = {}
    optimizers = {}
    loss_history = {}
    train_acc_history = {}
    test_acc_result = {}

    for wd in weight_decays:
        key = f"wd={wd}"
        networks[key] = MultiLayerNet(
            input_size=784,
            hidden_size_list=[100, 100, 100, 100],  # 4층 MLP
            output_size=10,
            weight_decay_lambda=wd,
        )
        optimizers[key] = Adam(lr=0.001)
        loss_history[key] = []
        train_acc_history[key] = []

    # 3. 학습 설정
    max_iterations = 2000
    batch_size = 128

    print("=== weight_decay_lambda 비교 실험 시작 ===")
    print(f"weight_decays = {weight_decays}")
    print(f"max_iterations = {max_iterations}, batch_size = {batch_size}")

    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for wd in weight_decays:
            key = f"wd={wd}"
            network = networks[key]
            optimizer = optimizers[key]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            loss = network.loss(x_batch, t_batch)
            loss_history[key].append(loss)

        # 200 iteration마다 로그/훈련 정확도 찍기
        if i % 200 == 0:
            print(f"========== iteration:{i} ==========")
            for wd in weight_decays:
                key = f"wd={wd}"
                network = networks[key]
                current_loss = loss_history[key][-1]
                train_acc = network.accuracy(x_train, y_train_int[:train_size])
                train_acc_history[key].append(train_acc)
                print(f"{key} | loss: {current_loss:.4f}, train_acc: {train_acc:.4f}")

    # 4. 최종 Test Accuracy
    print("\n=== 최종 Test Accuracy (weight decay 비교) ===")
    for wd in weight_decays:
        key = f"wd={wd}"
        network = networks[key]
        acc = network.accuracy(x_test, y_test_int)
        test_acc_result[key] = acc
        print(f"[{key}] Test Accuracy: {acc:.4f}")

    # 5. Loss 곡선 그래프
    plt.figure(figsize=(8, 5))
    iters = np.arange(max_iterations)
    for wd in weight_decays:
        key = f"wd={wd}"
        plt.plot(iters, loss_history[key], label=key)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Effect of Weight Decay (Train Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("weight_decay_compare_loss.png")
    print("Loss 그래프가 'weight_decay_compare_loss.png' 이름으로 저장되었습니다.")
    

    # 6. Test Accuracy 막대 그래프
    plt.figure(figsize=(6, 4))
    keys = list(test_acc_result.keys())
    values = [test_acc_result[k] for k in keys]
    plt.bar(keys, values)
    plt.ylim(0.0, 1.0)
    plt.xlabel("weight_decay_lambda")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Weight Decay")
    plt.tight_layout()
    plt.savefig("weight_decay_compare_accuracy.png")
    print("정확도 그래프가 'weight_decay_compare_accuracy.png' 이름으로 저장되었습니다.")
    
    plt.show()


if __name__ == "__main__":
    main()
