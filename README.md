# 신경망 조별과제

## Fashion-mnist 다운로드
https://github.com/zalandoresearch/fashion-mnist
* ZIP 파일 다운로드 후 압축 해제
* data/fashion/... (4개 파일)
* Neural_Network_Project/dataset 에 옮기기

## 실행


## 최고 Accuracy를 도출하는 방법 (PPT 활용 예정, 수정중)
* Train / Validation 분리
* BatchNorm / Dropout 적용한 MultiLayerNetExtend 사용
* 적절한 WeightDecay & Learning-rate schedule 적용
* Epoch 수 충분히 늘리기 (200~300 epoch)
* Mini-batch SGD + Adam 혼합 or AdamW 사용
* 성능 좋은 layer 구성 (128-128-64-64)
* EarlyStopping or Best model 저장

# Neural_Network_Project

> Python 기반 신경망 프로젝트 (Fashion-MNIST)
> Multi-layer Perceptron(MLP) / CNN 구조를 학습하고 평가합니다.
> Weight Decay, Dropout, BatchNorm 등 Regularization 기법 적용 가능

---

## 📂 프로젝트 구조 (수정중)

```
Neural_Network_Project/
 ├── dataset/                # Fashion-MNIST 데이터 저장 폴더
 ├── models/                 # MLP/CNN 모델 정의
 ├── utils/                  # 데이터 처리 및 유틸
 ├── train.py                # 학습 실행 스크립트
 ├── evaluate.py             # 테스트/검증 스크립트
 ├── plot.py                 # 손실/정확도 시각화
 └── README.md
```

* **dataset/**: Fashion-MNIST 4개 파일을 저장
* **models/**: MLP 또는 CNN 모델 클래스
* **utils/**: 데이터 로더, 전처리, 보조 함수
* **train.py**: 학습 메인 스크립트
* **evaluate.py**: 학습된 모델 성능 평가
* **plot.py**: 학습 곡선 시각화

---

## 🚀 설치 및 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/kangwoohyun999/Neural_Network_Project.git
cd Neural_Network_Project
```

### 2. Python 환경 구성

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install numpy matplotlib torch torchvision
```

### 3. 데이터 준비

Fashion-MNIST 데이터 ZIP 다운로드 후 압축 해제 → dataset 폴더에 4개 파일 복사:

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

### 4. 학습 실행

```bash
python train.py
```

### 5. 성능 평가

```bash
python evaluate.py
```

---

## 🧠 코드 동작 흐름 (train.py 기준)

1. **데이터 로딩**: Fashion-MNIST 불러오기 → Train/Validation 분리 → DataLoader 구성
2. **모델 생성**: MLP / CNN 모델 초기화 (He/Xavier Init 가능)
3. **손실 함수 & 옵티마이저**: CrossEntropyLoss + SGD/Adam/AdamW
4. **학습 반복**:

   * Forward → Loss 계산
   * Backward → Gradient 계산
   * Optimizer Step → 가중치 업데이트
   * Accuracy / Loss 기록
5. **검증**: Epoch마다 Validation accuracy 확인
6. **결과 저장**: 모델(.pth) 저장, plot.py로 학습 곡선 시각화

---

## ⚙️ Weight Decay (가중치 감소)

### 개념

* 모델 가중치 크기를 줄이는 regularization
* 과적합 방지 및 일반화 성능 향상

### 적용 방법 1: AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### 적용 방법 2: SGD + weight_decay

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### 학습률 스케줄링 (옵션)

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

---

## 📈 기타 권장 설정

| 항목                  | 설명               |
| ------------------- | ---------------- |
| Dropout             | 과적합 방지           |
| Batch Normalization | 학습 안정성 증가        |
| Xavier / He Init    | 깊은 네트워크 학습 속도 개선 |
| Early Stopping      | 불필요한 epoch 학습 방지 |
| Train/Val split     | 과적합 모니터링 필수      |

---

## 📜 예시 학습 코드 (Weight Decay 포함)

```python
import torch
from models.mlp import MLP
from utils.dataset import load_fashion_mnist

# 1. 데이터
train_loader, val_loader = load_fashion_mnist(batch_size=64)

# 2. 모델
model = MLP()
model.to("cuda")

# 3. Optimizer with Weight Decay
optimizer = torch.optim.A
```
