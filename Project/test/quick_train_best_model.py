# coding: utf-8
"""
최적 설정으로 빠르게 모델을 학습하는 스크립트
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pickle
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from utils.mnist_reader import load_fashion_mnist

# 데이터 로드
print("데이터 로딩 중...")
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=True, normalize=True)
print(f"Train 데이터: {x_train.shape}, Test 데이터: {x_test.shape}")

# ============================
# 추천 설정들 (경험적으로 좋은 성능을 보이는 조합)
# ============================

configurations = {
    '1_Adam_standard': {
        'hidden_size_list': [128, 128, 64],
        'weight_decay_lambda': 0.0001,
        'optimizer': 'Adam',
        'optimizer_param': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'batch_size': 100,
        'epochs': 30
    },
    '2_Adam_deeper': {
        'hidden_size_list': [256, 128, 64],
        'weight_decay_lambda': 0.0001,
        'optimizer': 'Adam',
        'optimizer_param': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'batch_size': 128,
        'epochs': 30
    },
    '3_RMSprop_standard': {
        'hidden_size_list': [128, 128, 64],
        'weight_decay_lambda': 0.0001,
        'optimizer': 'RMSprop',
        'optimizer_param': {'lr': 0.001, 'decay_rate': 0.99},
        'batch_size': 100,
        'epochs': 30
    },
    '4_Momentum_standard': {
        'hidden_size_list': [128, 128],
        'weight_decay_lambda': 0.0001,
        'optimizer': 'Momentum',
        'optimizer_param': {'lr': 0.01, 'momentum': 0.9},
        'batch_size': 100,
        'epochs': 30
    },
    '5_Adam_wide': {
        'hidden_size_list': [512, 256],
        'weight_decay_lambda': 0.0001,
        'optimizer': 'Adam',
        'optimizer_param': {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.999},
        'batch_size': 128,
        'epochs': 30
    },
}

# ============================
# 학습 실행
# ============================
results = {}
best_accuracy = 0
best_config_name = None

print("\n" + "=" * 70)
print("추천 설정으로 모델 학습 시작")
print("=" * 70)

for config_name, config in configurations.items():
    print(f"\n{'='*70}")
    print(f"설정: {config_name}")
    print(f"{'='*70}")
    print(f"Hidden Layers: {config['hidden_size_list']}")
    print(f"Weight Decay: {config['weight_decay_lambda']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Optimizer Params: {config['optimizer_param']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print("-" * 70)
    
    # 네트워크 생성
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=config['hidden_size_list'],
        output_size=10,
        activation='relu',
        weight_init_std='he',
        weight_decay_lambda=config['weight_decay_lambda']
    )
    
    # 학습
    trainer = Trainer(
        network, x_train, t_train, x_test, t_test,
        epochs=config['epochs'],
        mini_batch_size=config['batch_size'],
        optimizer=config['optimizer'],
        optimizer_param=config['optimizer_param'],
        evaluate_sample_num_per_epoch=1000,
        verbose=True
    )
    
    trainer.train()
    
    # 최종 평가
    final_test_acc = network.accuracy(x_test, t_test)
    final_train_acc = network.accuracy(x_train, t_train)
    
    print(f"\n최종 결과:")
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.4f}")
    
    # 결과 저장
    results[config_name] = {
        'config': config,
        'train_acc_list': trainer.train_acc_list,
        'test_acc_list': trainer.test_acc_list,
        'train_loss_list': trainer.train_loss_list,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'network': network
    }
    
    # 모델 저장
    model_filename = f"../data/network_{config_name}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(network, f)
    print(f"모델 저장: {model_filename}")
    
    # 최고 모델 추적
    if final_test_acc > best_accuracy:
        best_accuracy = final_test_acc
        best_config_name = config_name
        print(f"\n★★★ 새로운 최고 정확도: {best_accuracy:.4f} ★★★")
        
        # Team7 파일로도 저장
        with open('../data/network_Team7.pkl', 'wb') as f:
            pickle.dump(network, f)
        print("최고 모델을 network_Team7.pkl로 저장!")

# ============================
# 결과 요약
# ============================
print("\n" + "=" * 70)
print("학습 결과 요약")
print("=" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_acc'], reverse=True)

print(f"\n{'순위':<5} {'설정명':<25} {'Test Acc':<12} {'Train Acc':<12}")
print("-" * 70)
for rank, (name, result) in enumerate(sorted_results, 1):
    print(f"{rank:<5} {name:<25} {result['final_test_acc']:.4f}       {result['final_train_acc']:.4f}")

print(f"\n{'='*70}")
print(f"★ 최고 성능 모델: {best_config_name}")
print(f"★ Test Accuracy: {best_accuracy:.4f}")
print(f"★ 저장 위치: ../data/network_Team7.pkl")
print(f"{'='*70}")

# ============================
# 시각화
# ============================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Test Accuracy 비교
ax1 = axes[0, 0]
for name, result in sorted_results:
    epochs = np.arange(1, len(result['test_acc_list']) + 1)
    ax1.plot(epochs, result['test_acc_list'], marker='o', label=name, linewidth=2)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Train Loss 비교
ax2 = axes[0, 1]
for name, result in sorted_results:
    iterations = np.arange(1, len(result['train_loss_list']) + 1)
    # Loss가 너무 많으면 샘플링
    if len(iterations) > 1000:
        step = len(iterations) // 1000
        iterations = iterations[::step]
        loss_list = np.array(result['train_loss_list'])[::step]
    else:
        loss_list = result['train_loss_list']
    ax2.plot(iterations, loss_list, label=name, alpha=0.7)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Train Loss', fontsize=12)
ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 최종 정확도 막대 그래프
ax3 = axes[1, 0]
names = [name for name, _ in sorted_results]
test_accs = [result['final_test_acc'] for _, result in sorted_results]
colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
bars = ax3.barh(names, test_accs, color=colors, alpha=0.8, edgecolor='black')
ax3.set_xlabel('Test Accuracy', fontsize=12)
ax3.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)
for bar, acc in zip(bars, test_accs):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2., 
             f'{acc:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

# 4. 최고 모델의 Train vs Test Accuracy
ax4 = axes[1, 1]
if best_config_name:
    best_result = results[best_config_name]
    epochs = np.arange(1, len(best_result['test_acc_list']) + 1)
    ax4.plot(epochs, best_result['train_acc_list'], 
             marker='o', label='Train Accuracy', linewidth=2, color='blue')
    ax4.plot(epochs, best_result['test_acc_list'], 
             marker='s', label='Test Accuracy', linewidth=2, color='red')
    ax4.set_xlabel('Epochs', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title(f'Best Model: {best_config_name}', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.7, 1.0])

plt.tight_layout()
plt.savefig('../utils/training_results.png', dpi=300, bbox_inches='tight')
print(f"\n그래프 저장: ../utils/training_results.png")

# 결과 저장
with open('../data/training_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"실험 결과 저장: ../data/training_results.pkl")

plt.show()

print("\n학습 완료!")
print(f"최고 모델이 network_Team7.pkl로 저장되었습니다.")
print(f"Test Accuracy: {best_accuracy:.4f}")