# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# CNN(Convolution Neural Network, 합성곱 신경망)
# =============================================================================
# - 심층 신경망에 컨볼루션(합성곱) 장치를 추가한 신경망
# - 시각적 이미지를 분석하는데 사용
# - 합성곱 연산, 서브샘플링 연산, 드롭 아웃 기법 등이 추가로 적용
# - 속도, 과적합, 정확도 등에 대한 해결


# [ Convolution(합성곱) ]
# 인근에 있는 신호에 가중치를 부여하여 가중합의 신호로 축소시키는 과정 혹은 결과


# [ 마스크 ]
# - 컨볼루션시 사용되는 2차원 형태의 가중치 크기(2X2, 3X3, 4X4 등을 주로 사용)
# - 마스크 사이즈가 클수록 리턴되는 합성곱 사이즈가 작아짐
#  (차원축소 효과가 커서 총 학습속도를 줄임 but 너무 크면 의미성이 약한 신호가 리턴될 수 있음)


# [ 서브 샘플링(풀링) ]
# - 피처맵의 크기를 줄이는 기법 > 차원축소가 목적
# - 최대 풀(Max Pool), 최소 풀(Min Pool), 평균 풀(Average Pool) 연산 수행


# [ 드롭 아웃 ]
# 노드 off 비율 정함(0.5, 0.25)




# 예제) CNN을 사용한 mnist 손글씨 분류
# 1. 데이터 로딩
from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x.shape              # (60000, 28, 28)
test_x.shape               # (10000, 28, 28)



# 2. 데이터의 변환
# CNN에서는 4차원헝태 요구
train_x = train_x.reshape((60000, 28, 28, 1))
test_x = test_x.reshape((10000, 28, 28, 1))

# X 데이터 변환 : 스케일링
train_x = train_x / 255
test_x  = test_x / 255

# Y 데이터 변환 : 더미변수 변환
train_y_dm = pd.get_dummies(train_y).values
test_y_dm  = pd.get_dummies(test_y).values

train_y_dm.shape           # (60000, 10) 분리된 Y의 수 = 10
test_y_dm.shape            # (10000, 10) 분리된 Y의 수 = 10 (train/test 서로 일치)



# 3. 모델 정의
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, 
                 kernel_size = (3,3),          # mask size
                 input_shape = (28, 28, 1),    # 한 개의 데이터 기준 사이즈
                 activation = 'relu'))
model.add(Conv2D(64, 
                 kernel_size = (3,3),     
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=4))           # MaxPooling :  가장 큰 신호만 전달하는 기법
model.add(Dropout(0.25))                       # 25% 제거
model.add(Flatten())                           # 2차원 학습 > 1차 결과 리턴해야하므로 평탄화 필요
model.add(Dense(128, activation='relu'))       
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))



# 4. Compile
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = 'accuracy')



# 5. 정지규칙과 모델 저장
# 정지 규칙 생성
from keras.callbacks import EarlyStopping
m_stop = EarlyStopping(monitor='val_loss', patience=5)   

# 모델 저장
import os
modelpath = './model/mnist_cnn-{epoch:03d}-{val_loss:.4f}.hdf5'
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)



# 6. 훈련(fitting)
model.fit(train_x, train_y_dm, epochs = 2000, batch_size = 10, validation_split = 0.2,
          callbacks=[m_stop, checkpointer])



# 7. 평가
from keras.models import load_model, save_model
model = load_model('./model/mnist_cnn-005-0.0404.hdf5') # 각자 저장된 모델명 확인 후 입력

model.evaluate(test_x, test_y_dm)[1]



# 8. 예측 오류 데이터 확인
output = model.predict(test_x)
output.shape
output = np.round(output)            # 소수점 자리로 리턴되는 예측 결과를 
                                     # np.round를 통해 Y 형태(0 or 1)로 변환(반올림)
yhat = output.argmax(axis = 1)       # 실제 Y의 형태로 변환(각 관측치 최댓값 index 추출)

test_y == yhat                       # 실제값과 예측값 비교

import matplotlib.pyplot as plt
plt.imshow(test_x[test_y != yhat][0])# 예측 실패한 첫번째 값 시각화
test_x[test_y != yhat][0]
yhat[test_y != yhat][0]


plt.imshow(test_x[test_y != yhat][1])# 예측 실패한 두번째 값 시각화
test_x[test_y != yhat][1]            # 실제값 : 2
yhat[test_y != yhat][1]              # 예측값 : 7







# =============================================================================
# RNN(Recurrent Neural Network, 순환 신경망)
# =============================================================================
# - 은닉층에서 재귀적(다시 돌아감)인 신경망을 갖는 알고리즘
# - 음성 신호, 연속적 시계열 데이터 분석에 적합
# - 장기의존성문제(앞신호가 뒤로 전달 X 현상)로 인해 예측력 저하 발생 > LSTM으로 해결
























































