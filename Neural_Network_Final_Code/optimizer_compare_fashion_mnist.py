# optimizer_compare_fashion_mnist.py
# 다양한 Optimizer(SGD, Momentum, AdaGrad, Adam) 비교 실험 코드

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ====== (1) 경로 설정: common, utils 폴더 불러오기 ======
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD, Momentum, AdaGrad, Adam
from utils.mnist_reader import load_mnist


# ====== (2) Fashion-MNIST 로드 함수 ======
def load_fashion_mnist(normalize=True, one_hot_label=True):
    """
    Fashion-MNIST를 불러와서 (x_train, t_train), (x_test, t_test), (y_train, y_test) 형태로 리턴.
    x_* : (N, 784) float32
    t_* : one-hot (N, 10) or 레이블 (N,)
    """
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


def to_one_hot(y, num_classes=10):
    t = np.zeros((y.size, num_classes), dtype=np.float32)
    t[np.arange(y.size), y] = 1.0
    return t


# ====== (3) Optimizer 비교 실험 ======
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

    print(f"사용하는 train 데이터 개수: {train_size}")

    # 2. 비교할 Optimizer 설정
    optimizers = {
        'SGD': SGD(lr=0.01),
        'Momentum': Momentum(lr=0.01),
        'AdaGrad': AdaGrad(lr=0.01),
        'Adam': Adam(lr=0.001),   # Adam은 보통 조금 더 작은 lr 사용
    }

    # 3. Optimizer 마다 별도의 네트워크 생성
    networks = {}
    train_loss_history = {}

    for key in optimizers.keys():
        networks[key] = MultiLayerNet(
            input_size=784,
            hidden_size_list=[100, 100, 100, 100],  # 4층 MLP
            output_size=10,
        )
        train_loss_history[key] = []

    # 4. 학습 설정
    max_iterations = 2000       # iteration 수 (책 예제처럼)
    batch_size = 128

    print("=== Optimizer 비교 실험 시작 ===")
    print(f"max_iterations = {max_iterations}, batch_size = {batch_size}")

    for i in range(max_iterations):
        # 미니배치 뽑기
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 각 Optimizer에 대해 한 번씩 업데이트
        for key in optimizers.keys():
            network = networks[key]
            optimizer = optimizers[key]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            loss = network.loss(x_batch, t_batch)
            train_loss_history[key].append(loss)

        # 로그 출력 (100번마다)
        if i % 100 == 0:
            print(f"========== iteration:{i} ==========")
            for key in optimizers.keys():
                current_loss = train_loss_history[key][-1]
                print(f"{key}: {current_loss}")

    # 5. 최종 Test Accuracy 계산
    print("\n=== 최종 Test Accuracy ===")
    final_test_acc = {}
    for key in optimizers.keys():
        network = networks[key]
        acc = network.accuracy(x_test, y_test_int)
        final_test_acc[key] = acc
        print(f"[{key}] Test Accuracy: {acc:.4f}")

    # 6. Loss 곡선 그래프 그리기
    plt.figure(figsize=(8, 5))
    iters = np.arange(max_iterations)

    for key in optimizers.keys():
        plt.plot(iters, train_loss_history[key], label=key)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Optimizer Comparison on Fashion-MNIST (Train Loss)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 2.0)  # 필요에 따라 범위 조절
    plt.tight_layout()
    plt.savefig("optimizer_compare_loss.png")
    print("\nLoss 비교 그래프가 'optimizer_compare_loss.png' 이름으로 저장되었습니다.")
    

    # 7. Test Accuracy 막대 그래프 (선택)
    plt.figure(figsize=(6, 4))
    keys = list(final_test_acc.keys())
    values = [final_test_acc[k] for k in keys]
    plt.bar(keys, values)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Optimizer")
    plt.tight_layout()
    plt.savefig("optimizer_compare_accuracy.png")
    print("정확도 비교 그래프가 'optimizer_compare_accuracy.png' 이름으로 저장되었습니다.")
    
    plt.show()


if __name__ == "__main__":
    main()
