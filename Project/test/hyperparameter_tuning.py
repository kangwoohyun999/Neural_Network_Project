# coding: utf-8
"""
최적 하이퍼파라미터를 찾기 위한 그리드 서치 스크립트
특정 최적화 기법에 대해 learning rate, batch size 등을 탐색
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import product
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from run.train_fashion_mnist_team7 import load_fashion_mnist
from utils.mnist_reader import load_mnist


data_dir = os.path.join(os.path.dirname(__file__), 'data', 'fashion')
X_train, t_train = load_mnist(data_dir, kind='train')
X_test, t_test = load_mnist(data_dir, kind='t10k')

# ============================
# 하이퍼파라미터 그리드 설정
# ============================

# 실험할 최적화 기법 선택 (가장 유망한 것들)
target_optimizers = ['Adam', 'RMSprop', 'Momentum']

# Learning rate 범위
learning_rates = {
    'SGD': [0.001, 0.01, 0.05, 0.1, 0.5],
    'Momentum': [0.001, 0.01, 0.05, 0.1],
    'AdaGrad': [0.001, 0.01, 0.05, 0.1],
    'RMSprop': [0.0001, 0.001, 0.01, 0.1],
    'Adam': [0.0001, 0.0005, 0.001, 0.005, 0.01]
}

# Batch size 범위
batch_sizes = [32, 64, 100, 128, 256]

# 네트워크 구조 범위
hidden_layers = [
    [100, 100],
    [128, 128],
    [256, 128],
    [128, 128, 64],
    [256, 128, 64],
]

# Weight decay 범위
weight_decays = [0, 0.00001, 0.0001, 0.001]

# ============================
# 그리드 서치 실행
# ============================
max_epochs = 15
results = {}
best_accuracy = 0
best_config = None

print("=" * 70)
print("하이퍼파라미터 그리드 서치 시작")
print("=" * 70)

total_experiments = 0
for optimizer_name in target_optimizers:
    total_experiments += len(learning_rates[optimizer_name]) * len(batch_sizes) * len(hidden_layers) * len(weight_decays)

print(f"총 실험 수: {total_experiments}")
print()

experiment_count = 0

for optimizer_name in target_optimizers:
    print(f"\n{'='*70}")
    print(f"최적화 기법: {optimizer_name}")
    print(f"{'='*70}\n")
    
    for lr in learning_rates[optimizer_name]:
        for batch_size in batch_sizes:
            for hidden_list in hidden_layers:
                for weight_decay in weight_decays:
                    experiment_count += 1
                    
                    # 실험 이름
                    exp_name = f"{optimizer_name}_lr{lr}_bs{batch_size}_h{'-'.join(map(str, hidden_list))}_wd{weight_decay}"
                    
                    print(f"[{experiment_count}/{total_experiments}] {exp_name}")
                    
                    # 네트워크 생성
                    network = MultiLayerNet(
                        input_size=784,
                        hidden_size_list=hidden_list,
                        output_size=10,
                        activation='relu',
                        weight_init_std='he',
                        weight_decay_lambda=weight_decay
                    )
                    
                    # Optimizer 파라미터 설정
                    if optimizer_name == 'SGD':
                        opt_params = {'lr': lr}
                    elif optimizer_name == 'Momentum':
                        opt_params = {'lr': lr, 'momentum': 0.9}
                    elif optimizer_name == 'Nesterov':
                        opt_params = {'lr': lr, 'momentum': 0.9}
                    elif optimizer_name == 'AdaGrad':
                        opt_params = {'lr': lr}
                    elif optimizer_name == 'RMSprop':
                        opt_params = {'lr': lr, 'decay_rate': 0.99}
                    elif optimizer_name == 'Adam':
                        opt_params = {'lr': lr, 'beta1': 0.9, 'beta2': 0.999}
                    
                    # 학습
                    trainer = Trainer(
                        network, x_train, t_train, x_test, t_test,
                        epochs=max_epochs,
                        mini_batch_size=batch_size,
                        optimizer=optimizer_name,
                        optimizer_param=opt_params,
                        evaluate_sample_num_per_epoch=1000,
                        verbose=False
                    )
                    
                    try:
                        trainer.train()
                        
                        # 최종 정확도
                        final_test_acc = network.accuracy(x_test, t_test)
                        final_train_acc = network.accuracy(x_train, t_train)
                        
                        print(f"  Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}")
                        
                        # 결과 저장
                        results[exp_name] = {
                            'optimizer': optimizer_name,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'hidden_layers': hidden_list,
                            'weight_decay': weight_decay,
                            'train_acc_list': trainer.train_acc_list,
                            'test_acc_list': trainer.test_acc_list,
                            'final_train_acc': final_train_acc,
                            'final_test_acc': final_test_acc,
                        }
                        
                        # 최고 정확도 추적
                        if final_test_acc > best_accuracy:
                            best_accuracy = final_test_acc
                            best_config = exp_name
                            print(f"  ★★★ 새로운 최고 정확도: {best_accuracy:.4f} ★★★")
                            
                            # 최고 모델 저장
                            with open('../data/network_BEST.pkl', 'wb') as f:
                                pickle.dump(network, f)
                    
                    except Exception as e:
                        print(f"  ERROR: {str(e)}")
                        continue

# ============================
# 결과 분석
# ============================
print("\n" + "=" * 70)
print("그리드 서치 결과 분석")
print("=" * 70)

# 정확도 순으로 정렬
sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_acc'], reverse=True)

print(f"\n{'순위':<5} {'Test Acc':<12} {'실험 설정'}")
print("-" * 70)
for rank, (name, result) in enumerate(sorted_results[:20], 1):  # 상위 20개
    print(f"{rank:<5} {result['final_test_acc']:.4f}       {name}")

print(f"\n{'='*70}")
print(f"★ 최고 설정: {best_config}")
print(f"★ Test Accuracy: {best_accuracy:.4f}")

if best_config in results:
    best_result = results[best_config]
    print(f"\n최적 하이퍼파라미터:")
    print(f"  - Optimizer: {best_result['optimizer']}")
    print(f"  - Learning Rate: {best_result['learning_rate']}")
    print(f"  - Batch Size: {best_result['batch_size']}")
    print(f"  - Hidden Layers: {best_result['hidden_layers']}")
    print(f"  - Weight Decay: {best_result['weight_decay']}")
print(f"{'='*70}")

# ============================
# 시각화
# ============================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Learning Rate vs Accuracy (각 optimizer별)
ax1 = axes[0, 0]
for optimizer in target_optimizers:
    lr_acc = {}
    for name, result in results.items():
        if result['optimizer'] == optimizer:
            lr = result['learning_rate']
            if lr not in lr_acc:
                lr_acc[lr] = []
            lr_acc[lr].append(result['final_test_acc'])
    
    lrs = sorted(lr_acc.keys())
    avg_accs = [np.mean(lr_acc[lr]) for lr in lrs]
    ax1.plot(lrs, avg_accs, marker='o', label=optimizer, linewidth=2)

ax1.set_xlabel('Learning Rate', fontsize=12)
ax1.set_ylabel('Average Test Accuracy', fontsize=12)
ax1.set_title('Learning Rate vs Accuracy', fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Batch Size vs Accuracy
ax2 = axes[0, 1]
for optimizer in target_optimizers:
    bs_acc = {}
    for name, result in results.items():
        if result['optimizer'] == optimizer:
            bs = result['batch_size']
            if bs not in bs_acc:
                bs_acc[bs] = []
            bs_acc[bs].append(result['final_test_acc'])
    
    bss = sorted(bs_acc.keys())
    avg_accs = [np.mean(bs_acc[bs]) for bs in bss]
    ax2.plot(bss, avg_accs, marker='s', label=optimizer, linewidth=2)

ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_ylabel('Average Test Accuracy', fontsize=12)
ax2.set_title('Batch Size vs Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Weight Decay vs Accuracy
ax3 = axes[0, 2]
for optimizer in target_optimizers:
    wd_acc = {}
    for name, result in results.items():
        if result['optimizer'] == optimizer:
            wd = result['weight_decay']
            if wd not in wd_acc:
                wd_acc[wd] = []
            wd_acc[wd].append(result['final_test_acc'])
    
    wds = sorted(wd_acc.keys())
    avg_accs = [np.mean(wd_acc[wd]) for wd in wds]
    ax3.plot(wds, avg_accs, marker='^', label=optimizer, linewidth=2)

ax3.set_xlabel('Weight Decay', fontsize=12)
ax3.set_ylabel('Average Test Accuracy', fontsize=12)
ax3.set_title('Weight Decay vs Accuracy', fontsize=14, fontweight='bold')
ax3.set_xscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 상위 10개 모델의 학습 곡선
ax4 = axes[1, 0]
for name, result in sorted_results[:10]:
    epochs = np.arange(1, len(result['test_acc_list']) + 1)
    ax4.plot(epochs, result['test_acc_list'], marker='o', label=f"{result['optimizer']} (lr={result['learning_rate']})", alpha=0.7)
ax4.set_xlabel('Epochs', fontsize=12)
ax4.set_ylabel('Test Accuracy', fontsize=12)
ax4.set_title('Top 10 Models - Learning Curves', fontsize=14, fontweight='bold')
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(True, alpha=0.3)

# 5. Hidden Layer 구조 vs Accuracy
ax5 = axes[1, 1]
layer_acc = {}
for name, result in results.items():
    layer_str = '-'.join(map(str, result['hidden_layers']))
    if layer_str not in layer_acc:
        layer_acc[layer_str] = []
    layer_acc[layer_str].append(result['final_test_acc'])

layers = sorted(layer_acc.keys())
avg_accs = [np.mean(layer_acc[layer]) for layer in layers]
colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
bars = ax5.bar(range(len(layers)), avg_accs, color=colors, alpha=0.8, edgecolor='black')
ax5.set_xticks(range(len(layers)))
ax5.set_xticklabels(layers, rotation=45, ha='right')
ax5.set_ylabel('Average Test Accuracy', fontsize=12)
ax5.set_title('Network Architecture vs Accuracy', fontsize=14, fontweight='bold')
ax5.grid(True, axis='y', alpha=0.3)

# 6. 최고 모델의 학습 과정
ax6 = axes[1, 2]
if best_config in results:
    best_result = results[best_config]
    epochs = np.arange(1, len(best_result['test_acc_list']) + 1)
    ax6.plot(epochs, best_result['train_acc_list'], marker='o', label='Train Acc', linewidth=2, color='blue')
    ax6.plot(epochs, best_result['test_acc_list'], marker='s', label='Test Acc', linewidth=2, color='red')
    ax6.set_xlabel('Epochs', fontsize=12)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_title(f'Best Model: {best_config}', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../utils/hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
print(f"\n그래프 저장: ../utils/hyperparameter_tuning_results.png")

# 결과 저장
with open('../data/hyperparameter_tuning_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"실험 결과 저장: ../data/hyperparameter_tuning_results.pkl")

plt.show()

print("\n하이퍼파라미터 튜닝 완료!")