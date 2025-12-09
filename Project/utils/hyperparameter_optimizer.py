# coding: utf-8
import numpy as np
import itertools
from common.trainer import Trainer
import pickle

class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 클래스
    Grid Search와 Random Search를 지원합니다.
    """
    
    def __init__(self, network_builder, x_train, t_train, x_test, t_test):
        """
        Parameters
        ----------
        network_builder : function
            네트워크를 생성하는 함수 (하이퍼파라미터를 인자로 받음)
        x_train, t_train : 훈련 데이터
        x_test, t_test : 테스트 데이터
        """
        self.network_builder = network_builder
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.results = []
        
    def grid_search(self, param_grid, epochs=20, mini_batch_size=100, verbose=True):
        """Grid Search로 최적의 하이퍼파라미터 찾기
        
        Parameters
        ----------
        param_grid : dict
            탐색할 하이퍼파라미터 그리드
            예: {
                'lr': [0.001, 0.01, 0.1],
                'hidden_size': [50, 100, 200],
                'weight_decay_lambda': [0, 0.001, 0.01]
            }
        epochs : int
            학습 에포크 수
        mini_batch_size : int
            미니배치 크기
        verbose : bool
            진행 상황 출력 여부
            
        Returns
        -------
        best_params : dict
            최고 성능을 낸 하이퍼파라미터
        best_acc : float
            최고 정확도
        """
        # 모든 조합 생성
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        best_acc = 0
        best_params = None
        
        if verbose:
            print(f"=== Grid Search 시작: 총 {len(combinations)}개 조합 ===")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            if verbose:
                print(f"\n[{i+1}/{len(combinations)}] 테스트 중: {params}")
            
            # 네트워크 생성 및 학습
            acc, result = self._train_and_evaluate(params, epochs, mini_batch_size, verbose)
            
            # 결과 저장
            result['params'] = params
            result['test_acc'] = acc
            self.results.append(result)
            
            if acc > best_acc:
                best_acc = acc
                best_params = params
                if verbose:
                    print(f"★ 새로운 최고 정확도: {best_acc:.4f}")
        
        if verbose:
            print(f"\n=== Grid Search 완료 ===")
            print(f"최고 정확도: {best_acc:.4f}")
            print(f"최적 파라미터: {best_params}")
        
        return best_params, best_acc
    
    def random_search(self, param_distributions, n_iter=20, epochs=20, 
                     mini_batch_size=100, verbose=True, seed=None):
        """Random Search로 최적의 하이퍼파라미터 찾기
        
        Parameters
        ----------
        param_distributions : dict
            탐색할 하이퍼파라미터 분포
            예: {
                'lr': (0.0001, 0.1, 'log'),  # (min, max, scale)
                'hidden_size': (50, 300, 'int'),
                'weight_decay_lambda': (0, 0.1, 'uniform')
            }
        n_iter : int
            시도할 조합 수
        epochs : int
            학습 에포크 수
        mini_batch_size : int
            미니배치 크기
        verbose : bool
            진행 상황 출력 여부
        seed : int
            랜덤 시드
            
        Returns
        -------
        best_params : dict
            최고 성능을 낸 하이퍼파라미터
        best_acc : float
            최고 정확도
        """
        if seed is not None:
            np.random.seed(seed)
        
        best_acc = 0
        best_params = None
        
        if verbose:
            print(f"=== Random Search 시작: {n_iter}번 시도 ===")
        
        for i in range(n_iter):
            # 랜덤 파라미터 샘플링
            params = {}
            for key, (min_val, max_val, scale) in param_distributions.items():
                if scale == 'log':
                    params[key] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
                elif scale == 'int':
                    params[key] = int(np.random.uniform(min_val, max_val + 1))
                else:  # uniform
                    params[key] = np.random.uniform(min_val, max_val)
            
            if verbose:
                print(f"\n[{i+1}/{n_iter}] 테스트 중: {params}")
            
            # 네트워크 생성 및 학습
            acc, result = self._train_and_evaluate(params, epochs, mini_batch_size, verbose)
            
            # 결과 저장
            result['params'] = params
            result['test_acc'] = acc
            self.results.append(result)
            
            if acc > best_acc:
                best_acc = acc
                best_params = params
                if verbose:
                    print(f"★ 새로운 최고 정확도: {best_acc:.4f}")
        
        if verbose:
            print(f"\n=== Random Search 완료 ===")
            print(f"최고 정확도: {best_acc:.4f}")
            print(f"최적 파라미터: {best_params}")
        
        return best_params, best_acc
    
    def _train_and_evaluate(self, params, epochs, mini_batch_size, verbose):
        """주어진 파라미터로 네트워크를 학습하고 평가"""
        try:
            # 네트워크 생성
            network = self.network_builder(**params)
            
            # 학습
            trainer = Trainer(network, self.x_train, self.t_train, 
                            self.x_test, self.t_test,
                            epochs=epochs, mini_batch_size=mini_batch_size,
                            optimizer='Adam', optimizer_param={'lr': params.get('lr', 0.001)},
                            verbose=False)
            trainer.train()
            
            # 최종 테스트 정확도
            test_acc = network.accuracy(self.x_test, self.t_test)
            
            result = {
                'train_loss_list': trainer.train_loss_list,
                'train_acc_list': trainer.train_acc_list,
                'test_acc_list': trainer.test_acc_list
            }
            
            return test_acc, result
            
        except Exception as e:
            if verbose:
                print(f"오류 발생: {e}")
            return 0.0, {'train_loss_list': [], 'train_acc_list': [], 'test_acc_list': []}
    
    def save_results(self, filename='hyperparameter_results.pkl'):
        """결과를 파일로 저장"""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"결과가 {filename}에 저장되었습니다.")
    
    def get_top_n_results(self, n=5):
        """상위 n개 결과 반환"""
        sorted_results = sorted(self.results, key=lambda x: x['test_acc'], reverse=True)
        return sorted_results[:n]