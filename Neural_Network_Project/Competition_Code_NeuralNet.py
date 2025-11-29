# Competition_Code_NeuralNet.py

import argparse
import pickle
import numpy as np
from dataset.fashion_mnist_loader import load_data_from_pkl
import sys
import os

def evaluate(team, network_pkl, test_pkl):
    # load network
    if not os.path.exists(network_pkl):
        print("Network file not found:", network_pkl)
        sys.exit(1)

    # load test data (accepts (x_test,t_test) or full 4-tuple)
    x_train, t_train, x_test, t_test = load_data_from_pkl(test_pkl)
    if x_test is None:
        raise ValueError("test_pkl must contain test data.")

    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    if x_test.max() > 1.0:
        x_test /= 255.0

    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)

    preds = np.argmax(net.predict(x_test, train_flg=False), axis=1)
    if t_test.ndim != 1:
        t_labels = np.argmax(t_test, axis=1)
    else:
        t_labels = t_test

    acc = np.mean(preds == t_labels)
    print(f"Team {team} Accuracy = {acc:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=int, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    args = parser.parse_args()

    evaluate(args.team, args.network, args.test)
