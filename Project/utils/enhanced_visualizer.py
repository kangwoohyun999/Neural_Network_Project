# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class EnhancedVisualizer:
    """개선된 학습 과정 시각화 도구
    - 확대 기능
    - Smoothing 기능
    - 여러 실험 비교 기능
    - 고품질 그래프 저장
    """
    
    def __init__(self, figsize=(15, 5)):
        self.figsize = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        
    def smooth(self, data, window_size=10):
        """이동 평균을 사용한 데이터 스무딩
        
        Parameters
        ----------
        data : list or array
            스무딩할 데이터
        window_size : int
            이동 평균 윈도우 크기
            
        Returns
        -------
        smoothed_data : array
            스무딩된 데이터
        """
        if len(data) < window_size:
            return np.array(data)
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def plot_training_history(self, train_loss_list, train_acc_list, test_acc_list,
                              smooth_window=10, zoom_range=None, save_path=None):
        """학습 과정을 3개의 서브플롯으로 시각화
        
        Parameters
        ----------
        train_loss_list : list
            학습 손실 기록
        train_acc_list : list
            학습 정확도 기록
        test_acc_list : list
            테스트 정확도 기록
        smooth_window : int
            스무딩 윈도우 크기 (0이면 스무딩 안함)
        zoom_range : tuple
            확대할 범위 (start_epoch, end_epoch)
        save_path : str
            저장할 파일 경로 (None이면 저장 안함)
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # 손실 그래프
        x_loss = np.arange(len(train_loss_list))
        if smooth_window > 0:
            smoothed_loss = self.smooth(train_loss_list, smooth_window)
            axes[0].plot(x_loss, train_loss_list, alpha=0.3, color='blue', label='Original')
            axes[0].plot(x_loss, smoothed_loss, color='blue', linewidth=2, label=f'Smoothed (w={smooth_window})')
        else:
            axes[0].plot(x_loss, train_loss_list, color='blue', linewidth=2, label='Train Loss')
        
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 정확도 그래프
        epochs = np.arange(len(train_acc_list))
        axes[1].plot(epochs, train_acc_list, marker='o', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, test_acc_list, marker='s', label='Test Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Test Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 확대된 정확도 그래프
        if zoom_range is not None:
            start, end = zoom_range
            axes[2].plot(epochs[start:end], train_acc_list[start:end], 
                        marker='o', label='Train Accuracy', linewidth=2)
            axes[2].plot(epochs[start:end], test_acc_list[start:end], 
                        marker='s', label='Test Accuracy', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy')
            axes[2].set_title(f'Zoomed Accuracy (Epoch {start}-{end})')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            # Gap (과적합 정도) 표시
            gap = np.array(train_acc_list) - np.array(test_acc_list)
            axes[2].plot(epochs, gap, color='red', marker='x', linewidth=2)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy Gap')
            axes[2].set_title('Overfitting Gap (Train - Test)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
    
    def compare_experiments(self, experiments, metric='test_acc', smooth_window=0, save_path=None):
        """여러 실험 결과를 비교
        
        Parameters
        ----------
        experiments : list of dict
            각 실험의 데이터
            예: [
                {'name': 'Exp1', 'train_acc': [...], 'test_acc': [...]},
                {'name': 'Exp2', 'train_acc': [...], 'test_acc': [...]}
            ]
        metric : str
            비교할 메트릭 ('train_acc', 'test_acc', 'train_loss')
        smooth_window : int
            스무딩 윈도우 크기
        save_path : str
            저장할 파일 경로
        """
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        
        for i, exp in enumerate(experiments):
            name = exp.get('name', f'Experiment {i+1}')
            data = exp.get(metric, [])
            
            if len(data) == 0:
                continue
            
            x = np.arange(len(data))
            
            if smooth_window > 0:
                smoothed = self.smooth(data, smooth_window)
                plt.plot(x, data, alpha=0.2, color=colors[i])
                plt.plot(x, smoothed, label=name, linewidth=2, color=colors[i])
            else:
                plt.plot(x, data, label=name, linewidth=2, marker='o', 
                        markersize=4, color=colors[i])
        
        plt.xlabel('Epoch' if metric != 'train_loss' else 'Iteration')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison: {metric.replace("_", " ").title()}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"비교 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
    
    def plot_hyperparameter_analysis(self, results, param_name, save_path=None):
        """특정 하이퍼파라미터와 성능의 관계 시각화
        
        Parameters
        ----------
        results : list of dict
            하이퍼파라미터 최적화 결과
            각 dict는 'params'와 'test_acc' 키를 가져야 함
        param_name : str
            분석할 파라미터 이름
        save_path : str
            저장할 파일 경로
        """
        # 해당 파라미터 값과 정확도 추출
        param_values = []
        accuracies = []
        
        for result in results:
            if param_name in result['params']:
                param_values.append(result['params'][param_name])
                accuracies.append(result['test_acc'])
        
        if len(param_values) == 0:
            print(f"파라미터 '{param_name}'를 찾을 수 없습니다.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(param_values, accuracies, s=100, alpha=0.6, edgecolors='black')
        
        # 최고 성능 포인트 강조
        best_idx = np.argmax(accuracies)
        plt.scatter(param_values[best_idx], accuracies[best_idx], 
                   s=200, color='red', marker='*', 
                   label=f'Best: {param_values[best_idx]:.4f}', edgecolors='black')
        
        plt.xlabel(param_name)
        plt.ylabel('Test Accuracy')
        plt.title(f'Hyperparameter Analysis: {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"분석 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
    
    def plot_learning_rate_finder(self, lr_list, loss_list, save_path=None):
        """Learning Rate Finder 결과 시각화
        
        Parameters
        ----------
        lr_list : list
            학습률 리스트
        loss_list : list
            각 학습률에서의 손실값
        save_path : str
            저장할 파일 경로
        """
        plt.figure(figsize=(10, 6))
        plt.plot(lr_list, loss_list, linewidth=2)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, alpha=0.3)
        
        # 최소 손실 지점 표시
        min_idx = np.argmin(loss_list)
        plt.scatter(lr_list[min_idx], loss_list[min_idx], 
                   s=200, color='red', marker='*', 
                   label=f'Min Loss LR: {lr_list[min_idx]:.6f}', 
                   edgecolors='black', zorder=5)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning Rate Finder 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
    
    def plot_weight_distribution(self, network, save_path=None):
        """네트워크의 가중치 분포 시각화
        
        Parameters
        ----------
        network : MultiLayerNetExtend
            분석할 네트워크
        save_path : str
            저장할 파일 경로
        """
        # 가중치 추출
        weights = []
        layer_names = []
        
        for key in network.params.keys():
            if key.startswith('W'):
                weights.append(network.params[key].flatten())
                layer_names.append(key)
        
        # 서브플롯 생성
        n_layers = len(weights)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, (w, name) in enumerate(zip(weights, layer_names)):
            axes[i].hist(w, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{name} Distribution')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # 통계 정보 표시
            mean = np.mean(w)
            std = np.std(w)
            axes[i].axvline(mean, color='red', linestyle='--', 
                           label=f'Mean: {mean:.4f}')
            axes[i].axvline(mean + std, color='orange', linestyle='--', 
                           label=f'Std: {std:.4f}')
            axes[i].axvline(mean - std, color='orange', linestyle='--')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"가중치 분포 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()