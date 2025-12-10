# 신경망 조별과제

## pkl 파일은 예시 파일입니다. 생성된 파일 아닙니다.

## 실행 방법
ex) C:\Users\user\Downloads\Neural_Network_Project\dataset\mnist.py
* mnist.py 실행
* (테스트중)

## Fashion-mnist 다운로드 (mnist 실행하면 알아서 설치됨)
https://github.com/zalandoresearch/fashion-mnist
* ZIP 파일 다운로드 후 압축 해제
* data/fashion/... (4개 파일)
* Neural_Network_Project/dataset 에 옮기기

## 실행


## 🎯 간단 소개

1. 목적

* Fashion-MNIST 분류
* 교재 4–6장 기반
* Adam 사용
* 6층 이하 신경망

2. 데이터셋 설명

* 28×28 gray, 10 classes
* Train 60,000 / Test 10,000

3. 모델 구조

* Input 784
* Dense 256 → ReLU
* Dense 256 → ReLU
* Dense 128 → ReLU
* Output 10

4. 방법론

* Adam → 빠른 수렴
* Dropout → 과적합 완화
* ReLU + He 초기화
* 미니배치 학습

5. 실험

* Learning rate 비교
* Dropout 유무 비교
* Batchnorm 유무 비교

6. 결과

* 최종 Train/Test accuracy
* Loss / Accuracy curve

7. 결론

* Dropout이 가장 효과적
* Adam이 SGD 대비 빠름

---

## 📂 프로젝트 구조 (아래 전부 다 수정중)

```
Neural_Network_Project
 │
 ├── common
 │    ├─ __init__.py
 │    ├─ functions.py
 │    ├─ layers.py
 │    ├─ optimizer.py
 │    ├─ util.py
 │    ├─ multi_layer_net.py
 │    ├─ multi_layer_net_extend.py
 │    └─ gradient.py
 │
 ├─ data
 │    └─ fashion
 │          ├─ t10k-images-idx3-ubyte.gz
 │          ├─ t10k-labels-idx1-ubyte.gz
 │          ├─ train-images-idx3-ubyte.gz
 │          └─ train-labels-idx1-ubyte.gz
 │
 ├─ utils
 │    ├─ __init__.py
 │    ├─ argparser.py
 │    ├─ helper.py
 │    └─ mnist_reader.py
 │
 ├─ network_Team7.pkl
 ├─ activation_init_compare_fashion_mnist.py
 ├─ depth_compare_fashion_mnist.py
 ├─ dropout_compare_fashion_mnist.py
 ├─ weight_decay_compare_fashion_mnist.py
 ├─ optimizer_compare_fashion_mnist.py
 ├─ train_fashion_mnist_team7_final.py
 ├─ train_fashion_mnist_team7.py
 └─ README.md
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

## 맡은 역할 넣기
## ppt에 시행착오 과정, 출력 결과 넣기(layer별로, 모델별로)
## 하이퍼파라미터 최적화, 드롭아웃, 그래프 보기 쉽게(확대, smooth 등)
