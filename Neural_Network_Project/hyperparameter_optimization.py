import numpy as np
import pickle
import os
from dataset import load_data_from_pkl
from model import MultiLayerNet
from common.optimizer import Adam

TEAM_NUMBER = 7


# 랜덤 샘플링 함수
def sample_hyperparams():
    return {
        "hidden_sizes": np.random.choice([
            [512, 512, 256, 128],
            [512, 256, 128],
            [784, 512, 256, 128],
            [256, 256, 128],
            [1024, 512, 256]
        ]),
        "use_batchnorm": np.random.choice([True, False]),
        "use_dropout": np.random.choice([True, False]),
        "dropout_ratio": np.random.uniform(0.2, 0.5),
        "lr": 10 ** np.random.uniform(-4, -2),  # 1e-4 ~ 1e-2
        "batch_size": np.random.choice([64, 128, 256]),
        "epochs": 5  # Hyperopt 반복이므로 짧게
    }

# 학습 함수
def train(model, x_train, t_train, x_val, t_val, lr, batch_size, epochs):
    optimizer = Adam(lr=lr)
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    best_val_acc = 0
    best_params = None

    for epoch in range(epochs):
        idx = np.random.permutation(train_size)
        x_train = x_train[idx]
        t_train = t_train[idx]

        for i in range(iter_per_epoch):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_t = t_train[i * batch_size:(i + 1) * batch_size]

            grads = model.gradient(batch_x, batch_t)
            optimizer.update(model.params, grads)

        # epoch 마다 평가
        train_acc = model.accuracy(x_train, t_train)
        val_acc = model.accuracy(x_val, t_val)

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = pickle.dumps(model.params)  # deepcopy 대체

    # best 파라미터 복원
    if best_params is not None:
        model.params = pickle.loads(best_params)

    return best_val_acc

# 하이퍼파라미터 탐색 시작
def hyperparameter_search():
    train_data_path = "train_data.pkl"
    x_train, t_train, x_val, t_val = load_data_from_pkl(train_data_path)

    best_acc = 0
    best_setting = None
    best_model_params = None

    trials = 10  # 시도 횟수

    for trial in range(1, trials + 1):
        print(f"\n==============================")
        print(f"⚡ Trial {trial}/{trials}")
        print("==============================")

        params = sample_hyperparams()

        print("Sampled Hyperparams:")
        print(params)

        model = MultiLayerNet(
            input_size=784,
            hidden_sizes=params["hidden_sizes"],
            output_size=10,
            weight_init_std='he',
            use_batchnorm=params["use_batchnorm"],
            use_dropout=params["use_dropout"],
            dropout_ratio=params["dropout_ratio"]
        )

        val_acc = train(
            model,
            x_train, t_train,
            x_val, t_val,
            lr=params["lr"],
            batch_size=params["batch_size"],
            epochs=params["epochs"]
        )

        print(f"Trial {trial} Finished → Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_setting = params
            best_model_params = model.params.copy()

    print("\n=================================")
    print("🎉 Hyperparameter Search Finished")
    print("=================================")
    print("Best Validation Accuracy:", best_acc)
    print("Best Hyperparameters:", best_setting)

    # best 모델 저장
    model_path = f"network_Team{TEAM_NUMBER}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model_params, f)

    print(f"\n💾 Best Model Saved → {model_path}\n")


if __name__ == "__main__":
    hyperparameter_search()
