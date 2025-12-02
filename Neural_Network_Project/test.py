from data.mnist_reader import load_mnist

# 학습(train) 데이터 불러오기
x_train, t_train = load_mnist('dataset', kind='train')

# 테스트(test) 데이터 불러오기
x_test, t_test = load_mnist('dataset', kind='t10k')

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)