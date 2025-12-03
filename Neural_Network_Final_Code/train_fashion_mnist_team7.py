# train_fashion_mnist_team7.py
# 7조 Fashion-MNIST 팀플용 기본 학습 코드

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ====== (1) 경로 설정: common, utils 폴더 불러오기 ======
# 이 파일이 있는 위치 기준으로 상위 폴더를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# common 폴더에 있는 수업 코드들 임포트
from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam  # 필요하면 SGD, Momentum 등으로 바꿀 수 있음

# utils 폴더에 있는 Fashion-MNIST 로더 (교수님이 링크 준 mnist_reader.py)
from utils.mnist_reader import load_mnist


# ====== (2) 데이터 로드 & 전처리 ======
def load_fashion_mnist(normalize=True, one_hot_label=True):
    """
    Fashion-MNIST를 불러와서 (x_train, t_train), (x_test, t_test) 형태로 리턴.
    x_* : (N, 784) float32
    t_* : one-hot (N, 10) or 레이블 (N,)
    """
    # mnist_reader.py 기준 예시:
    # X_train, y_train = load_mnist('data/fashion', kind='train')
    # X_test, y_test = load_mnist('data/fashion', kind='t10k')

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
    """정수 레이블을 one-hot 벡터로 변환"""
    t = np.zeros((y.size, num_classes), dtype=np.float32)
    t[np.arange(y.size), y] = 1.0
    return t


# ====== (3) 학습 루프 ======
def train():
    # 1. 데이터 불러오기
    (x_train, t_train), (x_test, t_test), (y_train_int, y_test_int) = load_fashion_mnist(
        normalize=True,
        one_hot_label=True,
    )

    # 검증용 데이터 분리 (train 일부 떼어서 validation으로 사용)
    validation_size = 10000
    x_val = x_train[-validation_size:]
    t_val = t_train[-validation_size:]
    x_train_use = x_train[:-validation_size]
    t_train_use = t_train[:-validation_size]

    train_size = x_train_use.shape[0]

    # 2. 신경망 설정 (6층 이하 MLP, ReLU 사용)
    network = MultiLayerNet(
    input_size=784,
    hidden_size_list=[100, 100, 100, 100],
    output_size=10,
)


    # 3. Optimizer 설정 (6장 Adam 사용)
    learning_rate = 0.001
    optimizer = Adam(lr=learning_rate)

    # 4. 하이퍼파라미터 설정
    iters_num = 10000          # 전체 반복 횟수
    batch_size = 128
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    print("=== 학습 시작 ===")
    print(f"train size: {train_size}, batch size: {batch_size}, iter/epoch: {iter_per_epoch}")

    for i in range(iters_num):
        # (1) 미니배치 샘플링
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train_use[batch_mask]
        t_batch = t_train_use[batch_mask]

        # (2) 기울기 계산 (오차역전파)
        grads = network.gradient(x_batch, t_batch)

        # (3) 매개변수 갱신
        optimizer.update(network.params, grads)

        # (4) 손실 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # (5) 에폭마다 정확도 출력
        if (i + 1) % iter_per_epoch == 0:
            epoch = (i + 1) // iter_per_epoch
            train_acc = network.accuracy(x_train_use, np.argmax(t_train_use, axis=1))
            val_acc = network.accuracy(x_val, np.argmax(t_val, axis=1))
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            print(f"[Epoch {epoch:03d}] "
                  f"loss={loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # 최종 test accuracy
    test_acc = network.accuracy(x_test, np.argmax(t_test, axis=1))
    print("=== 학습 종료 ===")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # 5. Competition용 pkl 저장
    # 팀 번호에 맞게 파일명 설정 (여기서는 7조)
    pkl_path = os.path.join(os.path.dirname(__file__), "network_Team7.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(network, f)

    print(f"저장 완료: {pkl_path}")
    plot_graph(train_loss_list, train_acc_list, val_acc_list)


def plot_graph(train_loss_list, train_acc_list, val_acc_list):
    epochs = np.arange(1, len(train_acc_list) + 1)

    # Loss
    plt.figure()
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig("loss_graph.png")

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_acc_list, label="train_acc")
    plt.plot(epochs, val_acc_list, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_graph.png")

    plt.show()

if __name__ == "__main__":
    train()