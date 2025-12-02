import argparse
import pickle
import numpy as np
from dataset.fashion_mnist_loader import load_data_from_pkl
from model.multi_layer_net import MultiLayerNet
from common.optimizer import Adam
import os

def train_network(team, save_name, train_pkl,
                  epochs=30, batch_size=128, lr=0.001, early_stop_patience=5):
    x_train, t_train, x_val, t_val = load_data_from_pkl(train_pkl)
    if x_train is None:
        raise ValueError("train_pkl must include training data (x_train, t_train, x_val, t_val) for training.")

    # preprocess: reshape & normalize
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_val = x_val.reshape(x_val.shape[0], -1).astype(np.float32)
    if x_train.max() > 1.0:
        x_train /= 255.0
        x_val /= 255.0

    # build network (He init + BatchNorm + Dropout + Adam)
    net = MultiLayerNet(
        input_size=x_train.shape[1],
        hidden_sizes=[512, 512, 256, 128],  # 4 hidden layers => 총 6층 규정 준수
        output_size=10,
        weight_init_std='he',
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=0.3
    )

    optimizer = Adam(lr=lr)

    train_size = x_train.shape[0]
    iterations_per_epoch = max(train_size // batch_size, 1)

    best_val_acc = 0.0
    best_params = None
    no_improve = 0

    for epoch in range(1, epochs+1):
        idx = np.random.permutation(train_size)
        for i in range(0, train_size, batch_size):
            batch_idx = idx[i:i+batch_size]
            x_batch = x_train[batch_idx]
            t_batch = t_train[batch_idx]

            grads = net.gradient(x_batch, t_batch)
            optimizer.update(net.params, grads)

        train_acc = net.accuracy(x_train, t_train)
        val_acc = net.accuracy(x_val, t_val)
        val_loss = net.loss(x_val, t_val, train_flg=False)
        print(f"Epoch {epoch:02d}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}  Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in net.params.items()}
            no_improve = 0
            # save intermediate best (optional)
            with open(save_name + '.tmp', 'wb') as f:
                pickle.dump(net, f)
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # attach best params and save final pkl
    if best_params is not None:
        net.params = best_params

    with open(save_name, 'wb') as f:
        pickle.dump(net, f)
    # remove tmp if exists
    try:
        os.remove(save_name + '.tmp')
    except:
        pass

    print(f"Saved best model (val acc={best_val_acc:.4f}) to {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=int, required=True)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--train-pkl', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    save_name = args.save if args.save else f"network_Team{args.team}.pkl"
    train_network(args.team, save_name, args.train_pkl, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, early_stop_patience=args.patience)
