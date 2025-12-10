# 발표 대본
## Slide 1
안녕하십니까. 신경망 기말 발표를 시작하겠습니다. 팀 구성원으로는 한동윤, 강우현, 황지인입니다.

## 팀 역할
저는 초기 코드 제작과 PPT 제작, 발표를 담당하고, 한동윤님은 코드 완성과 발표, 황지인님은 코딩 서포트, 발표를 맡았습니다.

## 목차
1. 프로젝트 목적
2. 구조 및 모델 설계
3. 학습 과정
4. 주요 비교 실험 결과
5. 결론 및 향후 과제
6. 협업 과정

## 목적
동일한 MLP 구조에서 Optimizer, Weight Initialization(웨잇 이니셜라이제이션), Regularization(레귤러라이제이션)이 학습 성능에 미치는 영향을 분석하는 것이 목적입니다.

## 프로젝트 구조
공통 계층 모듈은 common 폴더, 데이터는 data 폴더, 스크립트는 기능별 파이썬 파일로 구성했습니다. 여러 폴더로 실험을 체계적으로 관리해봤습니다.

## 방법론 & 설계
모델은 MLP 기반이며, 입력은 28x28 이미지 -> 784차원 Flatten 처리, Hidden Layer는 ReLU 활성화 함수 적용, Output Layer는 Softmax 10 클래스입니다.

## 비교 실험 요소
* Optimizer : SGD, Momentum, AdaGrad, Adam
* Weight Initialization : Xavier(제이비어) vs He
