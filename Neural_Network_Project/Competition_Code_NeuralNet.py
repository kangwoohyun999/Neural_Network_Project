import os, sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle

# test_data.pkl 파일 불러오기
with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)

x_test = data["x_test"]
t_test = data["t_test"]

# x_test, t_test 변수 확인
print("x_test:", x_test.shape)
print("t_test:", t_test.shape)

# 자기 팀의 network 파일 불러오기
with open('network_Team7.pkl', 'rb') as f: # 1조에서 제출한 network 파일을 사용한 예시
    network = pickle.load(f)

accuracy = network.accuracy(x_test, t_test)
print("accuracy:", accuracy)