# coding: utf-8
"""
여러 최적화 기법을 비교하고 최고 정확도의 모델을 찾는 실험 스크립트
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
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=True, normalize=True)

# 학습 데이터 일부만 사용할 경우 (빠른 실험용)
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

# ============================
# 실험 설정
# ============================
max_epochs = 20
batch_size = 100

# 실험할 최적화 기법과 하이퍼파라미터 설정
optimizers = {
    'SGD': {'lr': 0.01},
    'SGD_fast': {'lr': 0.1},
    'Momentum': {'lr': 0.01, 'momentum': 0.9},
    'Momentum_fast': {'lr': 0.1, 'momentum': 0.9},
    'Nesterov': {'lr': 0.01, 'momentum': 0.9},
    'Nesterov_fast': {'lr': 0.1, 'momentum': 0.9},
    'AdaGrad': {'lr': 0.01},
    'AdaGrad_fast': {'lr': 0.1},
    'RMSprop': {'lr': 0.01, 'decay_rate': 0.99},
    'RMSprop_fast': {'lr': 0.001, 'decay_rate': 0.99},
    'Adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
    'Adam_fast': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999},
}

# 네트워크 설정 (여러 구조 실험 가능)
network_configs = [
    {
        'name': 'baseline',
        'hidden_size_list': [100, 100],
        'weight_decay_lambda': 0
    },
    {
        'name': 'deeper',
        'hidden_size_list': [128, 128, 64],
        'weight_decay_lambda': 0
    },
    {
        'name': 'wider',
        'hidden_size_list': [256, 128],
        'weight_decay_lambda': 0
    },
    {
        'name': 'with_decay',
        'hidden_size_list': [128, 128],
        'weight_decay_lambda': 0.0001
    },
]

# ============================
# 실험 실행
# ============================
results = {}
best_accuracy = 0
best_config = None

print("=" * 60)
print("최적화 기법 비교 실험 시작")
print("=" * 60)

for config in network_configs:
    config_name = config['name']
    print(f"\n{'='*60}")
    print(f"네트워크 구조: {config_name}")
    print(f"은닉층: {config['hidden_size_list']}")
    print(f"Weight Decay: {config['weight_decay_lambda']}")
    print(f"{'='*60}\n")
    
    for optimizer_name, optimizer_params in optimizers.items():
        experiment_name = f"{config_name}_{optimizer_name}"
        print(f"\n실험: {experiment_name}")
        print(f"Optimizer: {optimizer_name}, Params: {optimizer_params}")
        print("-" * 60)
        
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
            epochs=max_epochs,
            mini_batch_size=batch_size,
            optimizer=optimizer_name.split('_')[0],  # 'SGD_fast' -> 'SGD'
            optimizer_param=optimizer_params,
            evaluate_sample_num_per_epoch=1000,
            verbose=False
        )
        
        trainer.train()
        
        # 최종 정확도 평가
        final_test_acc = network.accuracy(x_test, t_test)
        final_train_acc = network.accuracy(x_train, t_train)
        
        print(f"최종 Train Accuracy: {final_train_acc:.4f}")
        print(f"최종 Test Accuracy: {final_test_acc:.4f}")
        
        # 결과 저장
        results[experiment_name] = {
            'network_config': config,
            'optimizer': optimizer_name,
            'optimizer_params': optimizer_params,
            'train_acc_list': trainer.train_acc_list,
            'test_acc_list': trainer.test_acc_list,
            'train_loss_list': trainer.train_loss_list,
            'final_train_acc': final_train_acc,
            'final_test_acc': final_test_acc,
            'network': network
        }
        
        # 최고 정확도 모델 추적
        if final_test_acc > best_accuracy:
            best_accuracy = final_test_acc
            best_config = experiment_name
            print(f"★ 새로운 최고 정확도! Test Acc: {best_accuracy:.4f}")
        
        # 모델 저장
        model_filename = f"../data/network_{experiment_name}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(network, f)
        print(f"모델 저장: {model_filename}")

# ============================
# 결과 분석 및 시각화
# ============================
print("\n" + "=" * 60)
print("실험 결과 요약")
print("=" * 60)

# 정확도 순으로 정렬
sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_acc'], reverse=True)

print(f"\n{'순위':<5} {'실험명':<35} {'Test Acc':<12} {'Train Acc':<12}")
print("-" * 70)
for rank, (name, result) in enumerate(sorted_results, 1):
    print(f"{rank:<5} {name:<35} {result['final_test_acc']:.4f}       {result['final_train_acc']:.4f}")

print(f"\n{'='*60}")
print(f"★ 최고 성능 모델: {best_config}")
print(f"★ Test Accuracy: {best_accuracy:.4f}")
print(f"{'='*60}")

# 그래프 그리기
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 최고 성능 모델들의 Test Accuracy 비교 (상위 6개)
ax1 = axes[0, 0]
top_n = 6
for name, result in sorted_results[:top_n]:
    epochs = np.arange(1, len(result['test_acc_list']) + 1)
    ax1.plot(epochs, result['test_acc_list'], marker='o', label=name, linewidth=2)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title(f'Top {top_n} Models - Test Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. 최고 성능 모델들의 Train Loss 비교
ax2 = axes[0, 1]
for name, result in sorted_results[:top_n]:
    iterations = np.arange(1, len(result['train_loss_list']) + 1)
    ax2.plot(iterations, result['train_loss_list'], label=name, alpha=0.7)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Train Loss', fontsize=12)
ax2.set_title(f'Top {top_n} Models - Training Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 최적화 기법별 평균 성능
ax3 = axes[1, 0]
optimizer_performance = {}
for name, result in results.items():
    opt = result['optimizer'].split('_')[0]
    if opt not in optimizer_performance:
        optimizer_performance[opt] = []
    optimizer_performance[opt].append(result['final_test_acc'])

avg_performance = {opt: np.mean(accs) for opt, accs in optimizer_performance.items()}
sorted_opts = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
opts, accs = zip(*sorted_opts)
colors = plt.cm.viridis(np.linspace(0, 1, len(opts)))
bars = ax3.bar(opts, accs, color=colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('Average Test Accuracy', fontsize=12)
ax3.set_title('Optimizer Performance Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_ylim([min(accs) - 0.02, max(accs) + 0.02])
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. 최고 vs 최저 성능 모델 비교
ax4 = axes[1, 1]
best_name, best_result = sorted_results[0]
worst_name, worst_result = sorted_results[-1]

epochs_best = np.arange(1, len(best_result['test_acc_list']) + 1)
epochs_worst = np.arange(1, len(worst_result['test_acc_list']) + 1)

ax4.plot(epochs_best, best_result['test_acc_list'], 
         marker='o', linewidth=3, label=f'Best: {best_name}', color='green')
ax4.plot(epochs_worst, worst_result['test_acc_list'], 
         marker='s', linewidth=3, label=f'Worst: {worst_name}', color='red')
ax4.set_xlabel('Epochs', fontsize=12)
ax4.set_ylabel('Test Accuracy', fontsize=12)
ax4.set_title('Best vs Worst Model Comparison', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../utils/optimizer_comparison_results.png', dpi=300, bbox_inches='tight')
print(f"\n그래프 저장: ../utils/optimizer_comparison_results.png")

# 결과를 파일로 저장
with open('../data/experiment_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"실험 결과 저장: ../data/experiment_results.pkl")

plt.show()

print("\n실험 완료!")