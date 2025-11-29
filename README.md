# 신경망 조별과제

## Fashion-mnist 다운로드
https://github.com/zalandoresearch/fashion-mnist
* ZIP 파일 다운로드 후 압축 해제
* data/fashion/... (4개 파일)
* Neural_Network_Project/dataset 에 옮기기

## 실행


## 최고 Accuracy를 도출하는 방법 (PPT 활용 예정)
* Train / Validation 분리
* BatchNorm / Dropout 적용한 MultiLayerNetExtend 사용
* 적절한 WeightDecay & Learning-rate schedule 적용
* Epoch 수 충분히 늘리기 (200~300 epoch)
* Mini-batch SGD + Adam 혼합 or AdamW 사용
* 성능 좋은 layer 구성 (128-128-64-64)
* EarlyStopping or Best model 저장
