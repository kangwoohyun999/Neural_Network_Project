# coding: utf-8
"""
하이퍼파라미터 최적화, 드롭아웃, 개선된 그래프 사용 예제
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from utils.mnist_reader import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from utils.hyperparameter_optimizer import HyperparameterOptimizer
from utils.enhanced_visualizer import EnhancedVisualizer

# ===========================================
# 예제 1: 드롭아웃을 사용한 기본 학습
# ===========================================
def example1_with_dropout():
    """드롭아웃을 사용한 학습 예제"""
    print("=" * 50)
    print("예제 1: 드롭아웃 사용 학습")
    print("=" * 50)
    
    # 데이터 로드
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 데이터 축소 (빠른 테스트를 위해)
    x_train = x_train[:1000]
    t_train = t_train[:1000]
    x_test = x_test[:300]
    t_test = t_test[:300]
    
    # 드롭아웃 있는 네트워크
    network_with_dropout = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100],
        output_size=10,
        activation='relu',
        weight_init_std='relu',
        weight_decay_lambda=0.01,
        use_dropout=True,      # 드롭아웃 사용
        dropout_ration=0.5     # 50% 드롭아웃
    )
    
    # 드롭아웃 없는 네트워크 (비교용)
    network_without_dropout = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100],
        output_size=10,
        activation='relu',
        weight_init_std='relu',
        weight_decay_lambda=0.01,
        use_dropout=False      # 드롭아웃 미사용
    )
    
    # 학습
    trainer_with = Trainer(network_with_dropout, x_train, t_train, x_test, t_test,
                          epochs=20, mini_batch_size=100,
                          optimizer='Adam', optimizer_param={'lr': 0.001},
                          verbose=True)
    trainer_with.train()
    
    trainer_without = Trainer(network_without_dropout, x_train, t_train, x_test, t_test,
                             epochs=20, mini_batch_size=100,
                             optimizer='Adam', optimizer_param={'lr': 0.001},
                             verbose=True)
    trainer_without.train()
    
    # 시각화
    visualizer = EnhancedVisualizer()
    
    # 비교 그래프
    experiments = [
        {
            'name': 'With Dropout (0.5)',
            'train_acc': trainer_with.train_acc_list,
            'test_acc': trainer_with.test_acc_list,
            'train_loss': trainer_with.train_loss_list
        },
        {
            'name': 'Without Dropout',
            'train_acc': trainer_without.train_acc_list,
            'test_acc': trainer_without.test_acc_list,
            'train_loss': trainer_without.train_loss_list
        }
    ]
    
    visualizer.compare_experiments(experiments, metric='test_acc', smooth_window=0,
                                  save_path='dropout_comparison.png')
    visualizer.compare_experiments(experiments, metric='train_loss', smooth_window=10,
                                  save_path='dropout_loss_comparison.png')


# ===========================================
# 예제 2: 하이퍼파라미터 그리드 서치
# ===========================================
def example2_grid_search():
    """Grid Search로 최적 하이퍼파라미터 찾기"""
    print("=" * 50)
    print("예제 2: Grid Search")
    print("=" * 50)
    
    # 데이터 로드
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 데이터 축소
    x_train = x_train[:1000]
    t_train = t_train[:1000]
    x_test = x_test[:300]
    t_test = t_test[:300]
    
    # 네트워크 빌더 함수
    def network_builder(lr=0.001, hidden_size=100, weight_decay_lambda=0.01, 
                       dropout_ratio=0.5):
        return MultiLayerNetExtend(
            input_size=784,
            hidden_size_list=[hidden_size, hidden_size],
            output_size=10,
            activation='relu',
            weight_init_std='relu',
            weight_decay_lambda=weight_decay_lambda,
            use_dropout=True,
            dropout_ration=dropout_ratio
        )
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'lr': [0.001, 0.005, 0.01],
        'hidden_size': [50, 100],
        'weight_decay_lambda': [0, 0.01],
        'dropout_ratio': [0.3, 0.5]
    }
    
    # Grid Search 실행
    optimizer = HyperparameterOptimizer(network_builder, x_train, t_train, x_test, t_test)
    best_params, best_acc = optimizer.grid_search(param_grid, epochs=10, verbose=True)
    
    # 결과 저장
    optimizer.save_results('grid_search_results.pkl')
    
    # 상위 5개 결과 출력
    top_results = optimizer.get_top_n_results(5)
    print("\n=== 상위 5개 결과 ===")
    for i, result in enumerate(top_results):
        print(f"{i+1}. 정확도: {result['test_acc']:.4f}, 파라미터: {result['params']}")
    
    # 시각화
    visualizer = EnhancedVisualizer()
    visualizer.plot_hyperparameter_analysis(optimizer.results, 'lr', 
                                           save_path='lr_analysis.png')
    visualizer.plot_hyperparameter_analysis(optimizer.results, 'dropout_ratio',
                                           save_path='dropout_analysis.png')


# ===========================================
# 예제 3: 랜덤 서치
# ===========================================
def example3_random_search():
    """Random Search로 최적 하이퍼파라미터 찾기"""
    print("=" * 50)
    print("예제 3: Random Search")
    print("=" * 50)
    
    # 데이터 로드
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 데이터 축소
    x_train = x_train[:1000]
    t_train = t_train[:1000]
    x_test = x_test[:300]
    t_test = t_test[:300]
    
    # 네트워크 빌더 함수
    def network_builder(lr=0.001, hidden_size=100, weight_decay_lambda=0.01, 
                       dropout_ratio=0.5):
        hidden_size = int(hidden_size)  # 정수로 변환
        return MultiLayerNetExtend(
            input_size=784,
            hidden_size_list=[hidden_size, hidden_size],
            output_size=10,
            activation='relu',
            weight_init_std='relu',
            weight_decay_lambda=weight_decay_lambda,
            use_dropout=True,
            dropout_ration=dropout_ratio
        )
    
    # 하이퍼파라미터 분포
    param_distributions = {
        'lr': (0.0001, 0.1, 'log'),              # 로그 스케일
        'hidden_size': (50, 200, 'int'),          # 정수
        'weight_decay_lambda': (0, 0.1, 'uniform'),  # 균등 분포
        'dropout_ratio': (0.2, 0.7, 'uniform')    # 균등 분포
    }
    
    # Random Search 실행
    optimizer = HyperparameterOptimizer(network_builder, x_train, t_train, x_test, t_test)
    best_params, best_acc = optimizer.random_search(param_distributions, n_iter=15, 
                                                    epochs=10, verbose=True, seed=42)
    
    # 결과 저장
    optimizer.save_results('random_search_results.pkl')
    
    # 상위 5개 결과 출력
    top_results = optimizer.get_top_n_results(5)
    print("\n=== 상위 5개 결과 ===")
    for i, result in enumerate(top_results):
        print(f"{i+1}. 정확도: {result['test_acc']:.4f}, 파라미터: {result['params']}")


# ===========================================
# 예제 4: 개선된 그래프 시각화
# ===========================================
def example4_enhanced_visualization():
    """개선된 시각화 기능 예제"""
    print("=" * 50)
    print("예제 4: 개선된 그래프 시각화")
    print("=" * 50)
    
    # 데이터 로드
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 데이터 축소
    x_train = x_train[:3000]
    t_train = t_train[:3000]
    x_test = x_test[:1000]
    t_test = t_test[:1000]
    
    # 네트워크 생성
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100],
        output_size=10,
        activation='relu',
        weight_init_std='relu',
        weight_decay_lambda=0.01,
        use_dropout=True,
        dropout_ration=0.5
    )
    
    # 학습
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                     epochs=30, mini_batch_size=100,
                     optimizer='Adam', optimizer_param={'lr': 0.001},
                     verbose=True)
    trainer.train()
    
    # 시각화
    visualizer = EnhancedVisualizer()
    
    # 1. 기본 그래프 (스무딩 적용)
    visualizer.plot_training_history(
        trainer.train_loss_list,
        trainer.train_acc_list,
        trainer.test_acc_list,
        smooth_window=20,
        save_path='training_history_smoothed.png'
    )
    
    # 2. 확대된 그래프 (마지막 10 에포크만)
    visualizer.plot_training_history(
        trainer.train_loss_list,
        trainer.train_acc_list,
        trainer.test_acc_list,
        smooth_window=10,
        zoom_range=(20, 30),
        save_path='training_history_zoomed.png'
    )


# ===========================================
# 예제 5: 배치 정규화 비교
# ===========================================
def example5_batch_normalization():
    """배치 정규화 효과 비교"""
    print("=" * 50)
    print("예제 5: 배치 정규화 비교")
    print("=" * 50)
    
    # 데이터 로드
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 데이터 축소
    x_train = x_train[:1000]
    t_train = t_train[:1000]
    x_test = x_test[:300]
    t_test = t_test[:300]
    
    # 배치 정규화 O
    network_with_bn = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100],
        output_size=10,
        activation='relu',
        weight_init_std='relu',
        use_batchnorm=True
    )
    
    # 배치 정규화 X
    network_without_bn = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100],
        output_size=10,
        activation='relu',
        weight_init_std='relu',
        use_batchnorm=False
    )
    
    # 학습
    trainer_with_bn = Trainer(network_with_bn, x_train, t_train, x_test, t_test,
                              epochs=20, mini_batch_size=100,
                              optimizer='Adam', optimizer_param={'lr': 0.001},
                              verbose=True)
    trainer_with_bn.train()
    
    trainer_without_bn = Trainer(network_without_bn, x_train, t_train, x_test, t_test,
                                 epochs=20, mini_batch_size=100,
                                 optimizer='Adam', optimizer_param={'lr': 0.001},
                                 verbose=True)
    trainer_without_bn.train()
    
    # 비교
    visualizer = EnhancedVisualizer()
    experiments = [
        {
            'name': 'With Batch Normalization',
            'test_acc': trainer_with_bn.test_acc_list
        },
        {
            'name': 'Without Batch Normalization',
            'test_acc': trainer_without_bn.test_acc_list
        }
    ]
    
    visualizer.compare_experiments(experiments, metric='test_acc',
                                  save_path='batch_norm_comparison.png')


if __name__ == '__main__':
    # 원하는 예제 실행
    print("실행할 예제를 선택하세요:")
    print("1. 드롭아웃 비교")
    print("2. Grid Search")
    print("3. Random Search")
    print("4. 개선된 시각화")
    print("5. 배치 정규화 비교")
    
    choice = input("선택 (1-5): ")
    
    if choice == '1':
        example1_with_dropout()
    elif choice == '2':
        example2_grid_search()
    elif choice == '3':
        example3_random_search()
    elif choice == '4':
        example4_enhanced_visualization()
    elif choice == '5':
        example5_batch_normalization()
    else:
        print("잘못된 선택입니다. 예제 1을 실행합니다.")
        example1_with_dropout()