
#
# 여러분은,
# 다음과 같이 test_data.pkl 파일을 불러와서
#


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
with open('network_Team1.pkl', 'rb') as f: # 1조에서 제출한 network 파일을 사용한 예시
    network = pickle.load(f)

# 자기 팀의 accuray 구하여 출력하기
accuracy = network.accuracy(x_test, t_test)
print("accuracy:", accuracy)

#
# network_Team1.pkl은 
# 1조(Team1)가 MultiLayerNet 객체 또는  MultiLayerNetExtend 객체를 
# network라고 명명하여 pkl 파일로 저장한 것임.
#

# trainer의 network를 저장하는 코드 예시는 다음과 같습니다. 
#     with open('network_Team1.pkl', 'wb') as f:
#        pickle.dump(trainer.network, f)
# 여러분은 적절하게게 network를 저장하시면 됩니다.


# 문의 사항이 있으면, 
# 즉시 경천관 404호(031-280-3661)로 오거나 저에게 전화해서 물어보시기 바랍니다.

# 15주 수업전에 팀 network.pkl 파일을 게시판에 제출하시기 바랍니다. 늦은 제출 감점있을 수 있음.
# 15주 수업시간에 최종발표를 모두 마치고, 
# test_data.pkl 파일 공개하고,
# 팀별로, Competition_Code.py 파일을 실행하여 accuracy 출력 값을 출력함. 

# 모두 화이팅!
