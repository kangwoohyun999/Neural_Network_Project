#
# 저는.
# 아래 코드를 참고하여, 
# 10개의 데이터를 랜덤 추출하여 data라는 딕셔너리 변수에 저장하여,
# test_data.pkl 파일을 드리겠습니다.
#

import os, sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

from common.util import shuffle_dataset
from data.mnist_reader import load_mnist # fashion minst data 로드
import pickle

x_test, t_test = load_mnist('../data/fashion', kind='t10k')

x_test, t_test = shuffle_dataset(x_test, t_test) # 순서 뒤섞기

num = 10

x_test = x_test[:num]
t_test = t_test[:num]
print(x_test.shape)
print(t_test.shape)

# 두 배열을 딕셔너리에 저장
data = {"x_test": x_test, "t_test": t_test}

# pickle로 저장
with open('test_data.pkl', 'wb') as f:
    pickle.dump(data, f)

#
# 여러분은 다음과 같이 test_data.pkl 파일을 불러 올 수 있습니다.
#

# pickle로 불러오기
with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)

# x_test, t_test에 저장
x_test = data["x_test"]
t_test = data["t_test"]

# x_test, t_test 크기 확인
print("test_data.pkl 파일 확인:")
print("x_test:", x_test.shape)
print("t_test:", t_test.shape)

#
# 첨부하는 Competition_Code_Team1.py 파일을 꼭 확인하세요
#