# -*- coding: utf-8 -*-

# =============================================================================
# 인공 신경망(ANN)
# =============================================================================
# - Deep Learning
# - 두뇌 신경세포인 뉴런에 전기 신호를 전달하는 모습을 모방한 기계학습 모델
# - 전기신호(입력데이터)를 연속적으로 보다 의미있는 신호 가공 과정을 반복
# - 입력값을 받아 출력값을 만들기 위해 활성함수 사용



# [ 용어 정리 ]
# 1) layer(층) 
#  : - 노드(뉴런)들이 있는 각 단계
#    - 특정 노드의 신호는 이전 층의 노드 신호들의 가중합
# 2) input layer(입력층) : 최초 신호를 받아들이는 레이어(단일층, 생략 불가)
# 3) hidden layer(은닉층) : 중간에 신호를 가공하는 층(여러 층일 수 있음)
# 4) output layer(출력층) : 예측값을 리턴하는 최종 레이어(단일층, 생략 불가)
# 5) perceptron(퍼셉트론) : 가장 기본적인 신경망 단위(input - hidden(1) - output)
# 6) activation function(활성함수) 
#  : - 목적 1) 노드의 신호를 다음 노드로 전달할지 여부를 결정(전달 or 전달X 결정)
#    - 목적 2) 이중분류 문제(0 or 1로 전달)
#    - S자 모양의 시그모이드, 렐루, 소프트맥스 함수 등



# [ 매개변수 / 초매개변수 ]
# - Parameter       : 가중치, 활성함수의 계수
# - Hyper-Parameter : 노드(유닛)의 수(단, 입력층/출력층 노드 수는 지정 불가), 층의 수,
#                     학습률(learning rate), batch_size

# input layer의 노드 수는 1개 이상 가능
# 회귀모델 output layer 노드 수 - 반드시 1개
# 분류모델 output layer 노드 수 - class 개수에 따라 달라짐
# ex) 2개 class : 출력층 노드 1개 or 2개
#     3개 class : 출력층 노드 1개 or 2개 or 3개(선호 : 3개)


# AND 연산  : 입력값(X, Y)이 모두 1이면 1 리턴, 그 외 0 리턴
# OR 연산   : 입력값(X, Y)이 모두 0이면 0 리턴, 그 외 1 리턴
# XOR  연산 : 입력값(X, Y)이 모두 같으면 0, 서로 다르면 1 리턴




#[ 신경망의 원리 ]
# 1. 가중치의 결정 : 경험적 결정 > 오차를 최소화하는 가중치 결정

# 2. 단층신경망에서의 가중치 결정 과정(델타 규칙)
#    - 델타 규칙에 의하면 어떤 입력 노드가 출력 노드의 오차에 기여했다면 
#      각 노드의 연결 가중치(w)는 해당 입력노드의 신호(출력)와 출력 노드의 오차에 비례
#      α(learning rate, 학습률) : 가중치를 얼마나 바꿀지를 결정하는 매개변수
#                                너무 크면 수렴 구간을 지나쳐 최적화 불가
#                                너무 작으면 수렴점을 찾기까지 시간이 많이 소요됨

# 3. 최적화
# 경사하강법(Gradient Descent) : 
# - 오차를 최소화 시키는 가중치 결정 단계
# - 최초 가중치를 임의로 설정
# - 임의로 설정된 가중치에 의해 오차 결정
# - 가중치 업데이트 > 변화에 따라 최적화 방식이 달라짐
# - 변경된 가중치에 의해 다시 오차 결정
# - 위 과정을 반복하여 오차를 최소화시키는 기울기 결정(미분 활용)


# 1) SGD(Stochastic Gradient Descent, 확률 경사 하강법)
#    - 하나의 학습 데이터마다 오차 계산, 이를 통해 가중치 업데이트
#      ex) 150개의 학습데이터일 경우 150번 업데이트
#          업데이트 화정이 매우 들쭉날쭉함 > 확률적  
       
# 2) batch(배치)
#    - 모든 데이터를 한 번 학습시켜서 1번의 가중치를 갱신
#    - 각 데이터를 학습시킬때마다 가중치를 얻고 이의 평균으로 최종 가중치를 갱신
#    - 모든데이터를 1번 학습 시켜서 1번의 가중치를 갱신하므로 갱신 속도가 느림
 
# 3) mini batch(미니 배치)
#    - 임의의 n개 데이터만 선택하여 가중치를 업데이트
#    - SGD + batch 혼합형
#      ex) 150개의 학습데이터, 50(batch size)개 샘플링 
#          > 3번 업데이트(mini batch(분리된 집합수)수와 같음)
#            batch size가 클수록 총 학습시간은 줄어듦
#            batch size = 1, SGD 동일



# 단층신경망(델타규칙)으로는 XOR 선형분리 불가 > 다층 신경망을 통해 해결
# 은닉층의 오차를 정의 불가 
# - 입력층에서 중간층으로 향하는 기울기 최적화 불가
# - 오차 역전파를 통해 해결 
#   : 오차를 결정하는 요인에 의해 마지막 층에서의 오차를 중간층으로 거꾸로 전달



# [ 반복과 관련된 용어 정리 ]
# - batch_size : 전체 데이터를 임의로 n개 샘플링하여 학습, 가중치 업데이트할 때 n의 값
#                (값의 범위 : 1 ~ 전체 데이터 수)

# - epoch      : 전체 데이터 셋 기준으로 신경망을 통과한 횟수
#                순전파 + 역전파가 이루어진 횟수
#                100 epoch > 100번의 순전파, 100번의 역전파

# - iteration  : 1-epoch를 마치는데 필요한 mini batch 수
#                즉 한 번의 epoch를 통해 발생한 가중치 업데이트 
#                700개 데이터를 100(batch_size)개씩 7개의 미니배치로 나눌 경우
#                1번의 epoch가 발생하게 되면 7번 업데이트 iteration = 7




# [ 활성화 함수 종류 ]
# 1. step function : [0,1]
# 2. sign function : [-1,1]
# 3. sigmoid function : [0,1]
# 4. linear function : [-inf, inf]
# 5. relu function : [0, inf]
# 6. tanh function : [-1,1]
# 7. softmax function : [0, 1]
#    - 다중클래스 분류 문제 시 마지막 층에서 사용
#    - 가중합의 크기를 총 합이 1인 새로운 가중치를 만드는 함수
#    - 가장 큰 신호에 가장 큰 가중치, 가중치가 가장 큰 노드 1, 나머지 0 신호로 변환

# ex) A, B, C class를 갖는 분류과제 적용 시 마지막층 설계
#     case 1 : 3개 노드 생성, sigmoid
#     case 2 : 3개 노드 생성, softmax




# [ 오차 함수(손실함수) 정의 및 종류 ]
# 오차와 기울기와의 관계를 활용하여 최적화 진행 > 최종 기울기 결정
# 1. 회귀 : MSE(Mean Squared Error), RMSE
# 2. 분류 : cross entropy
#          (binary_crossentropy - 이진, categorical_corssentropy - 다항)




# [ 기타 최적화 과정]
# 1. SGD(Stochastic Gradient Descent, 확률적 경사 하강법)
# 2. 모멘텀(Momentum)
# 3. 아다그라드(AdaGrad)
# 4. 아담(Adam) : 모멘텀 방식과 아다그라드 방식의 장점을 합친 알고리




# [ 과대적합 해결 방법 ]
# - epoch 수 조절(stopping rule 추가)
# - dataset 증가(학습 데이터 수를 늘림)
# - hidden layer 수 조절법
# - layer 구성 노드 수 조절
# - dropout : 신경망 모델의 학습 과정에서 신경망 일부를 사용하지 않는 방법
#             초기 드롭아웃, 공간적 드롭아웃, 시간적 드롭아웃




# [ 실습 ]
# pip install tensorflow
# pip install keras
import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# 1. 데이터 가공하기
#    문자 > 숫자 변환(label encoding)
s1 = Series(['A', 'B', 'B', 'A'])

#    1) 직접 변경
s1.replace({'A':0, 'B':1})


#    2) LabelEncoder
from sklearn.preprocessing import LabelEncoder
m_le = LabelEncoder()
m_le.fit(s1)                      # 입력된 dataset에 어떤 unique value가 있는지 파악
m_le.transform(s1)                # unqiue value에 맞게 각 data value 변환


#    3) ord
ord('a')                          # 문자마다 고유의 식별자 존재
ord(s1)                           # 단점 : 벡터연산 불가
s1.map(lambda x : ord(x))         # mapping 혹은 반복문을 통해 연산 적용



# 2. Y 변환(dummy 변수 변환)
y = [0, 1, 1, 2, 2]

#    1) pd.get_dummies
pd.get_dummies(y)                 # 기본적으로 y 개수만큼 분리
pd.get_dummies(y, prefix='y')     # y_0, y_1, y_2
pd.get_dummies(y, drop_first=True)# class수-1개만큼 분리
pd.get_dummies(y).values          # array 리턴


#    2) keras.untils.np_utils
from keras.utils import np_utils
np_utils.to_categorical(y)        # array 리턴 



# 예제) ANN model 적용한 iris data 분류
# 1. 데이터 로딩
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']


# 2. Y 변환
iris_y_dummy = pd.get_dummies(iris_y).values# array 형태가 속도적 면에서 유리할 수 있음


# 3. train/test 분리
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y_dummy, random_state = 0)


# 4. modeling
import tensorflow
import keras

from keras.models import Sequential # 여러층 설계를 위한 함수
from keras.layers.core import Dense # 한개 층(노드)을 구성하기 위한 함수

#    1) 모델 정의
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

model = Sequential()
model.add(Dense(units,              # 각 층의 노드(유닛) 수
                activation,         # 활성함수
                input_dim,          # input layer의 노드(유닛) 수
                ...)
          
model.add(Dense(30, activation = 'relu', input_dim = 4))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
          

#    2) 오차 함수, 최적화 함수 정의
model.compile(loss = 'mean_squared_error', # 분류과제 MSE 사용 이유 > 오분류율로 해석
              optimizer = 'adam',
              metrics = 'accuracy')


#    3) 학습
model.fit(train_x, train_y, epochs = 500, batch_size = 10)


#    4) 평가
#       - complie 속성에 따라 리턴 값이 달라짐
#       - 해당 모델의 경우 MSE, accuracy가 순서대로 보여짐
model.evaluate(test_x, test_y)[0]          # loss : 0.0193
model.evaluate(test_x, test_y)[1]          # accuracy : 0.9736




# [ 연습 문제 ] cancer data ANN 분류
# 1. 데이터 로딩
cancer = pd.read_csv('data/cancer.csv')
cancer_x = cancer.iloc[:, 2:]
cancer_y = cancer['diagnosis']


# 2. Y 변환
cancer_y_dummy = pd.get_dummies(cancer_y).values


# 3. scaling
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
cancer_x_sc = m_sc.fit_transform(cancer_x)


# 4. train/test 분리
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(cancer_x_sc, cancer_y_dummy, random_state = 0)


# 5. modeling
model2 = Sequential()
model2.add(Dense(15, activation = 'relu', input_dim = 30))
model2.add(Dense(7, activation = 'relu'))
model2.add(Dense(2, activation = 'softmax'))

model2.compile(loss = 'binary_crossentropy',
               optimizer = 'adam',
               metrics = 'accuracy')

model2.fit(train_x, train_y, epochs = 500, batch_size = 50)


# 6. 평가
model2.evaluate(test_x, test_y)[0]         # binary_crossentropy : 0.3193
model2.evaluate(test_x, test_y)[1]         # accuracy : 0.9580




# [ stopping rule 적용 및 모델 저장 ]
# 1. stopping rule
#    - 과적합을 방지하기 위한 정지 규칙
#    - 너무 많은 반복을 하게 될 경우 과적합 발생 > 적절한 수준에서 학습 진행 중지
#    - loss, metrics 등의 변화가 없을 때 주로 정지 규칙 정의

from keras.callbacks import EarlyStopping
m_stop = EarlyStopping(monitor='val_loss', # 관찰대상
                       patient=10)         # 중지 기준(10회 이상 변화가 없을 시 중단)


# 2. 모델 저장
#    - 모델 학습 결과는 일반적으로 메모리에 저장 > 휘발성, 손실 가능성 있음
#    - 디스크의 영역에 학습 결과 저장 > 영구 저장이 목적
import os 
MODEL_DIR = './model/'

# model을 저장할 directory가 없을 경우 생성
if not os.path.exists(MODEL_DIR) :
    os.mkdir(MODEL_DIR)

modelpath = './model/{epch:03d}-{val_loss:.4f}.hdf5'  
from keras.callbacks import ModelCheckpoint  


checkpointer = ModelCheckpoint(filepath = modelpath,      # 저장 형식
                               monitor = 'val_loss',      # 관찰 대상
                               savel_bes_only=True)       # 향상된 모델에 대해서만 저장


# 3. 정지 규칙을 적용한 학습(모델 저장)
model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              metrics='accuracy',
              optimizer='adam')

model.fit(train_x, train_y, epochs=2000, batch_size = 50,
          callbacks=[m_stop, checkpointer],
          validation_split=0.2)
    

# 4. model 평가
model.evaluate(test_x, test_y)[1]




# [ 연습문제 ] ANN을 이용한 얼굴 데이터 분류
# 1. 데이터 가공
#    1) X
#       - 하나의 데이터 기준 1차원 형식만 학습 가능
#       - 변수 스케일링 > X / 255 (RGB 값을 나누어 minmax scaling과 같은 효과)
#    2) Y
#       - 더미 변수로 변경

# 데이터 수집
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person = 50)

people_x = people['data']
people_x.shape                       # (1560, 2914)

people_y = people['target']
people_y.shape                       # (1560, )


# 데이터 가공
# 1) 스케일링
people_x = people_x / 255            # min-max scaling 효과
# 2) NN 학습을 위한 y 변수 변환
people_y_dummy = pd.get_dummies(people_y).values


# 2. 데이터 분리
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y, train_y_dm, test_y_dm = train_test_split(people_x, 
                                                                           people_y,
                                                                           people_y_dummy,
                                                                           random_state=0)


# 3. ANN 모델링
people['images'].shape                # 3차원 데이터 제공 (1560, 62, 47) > 원본 (CNN)
people['data'].shape                  # (1560, 2914) > 하나의 데이터 기준 1차원 평탄화 (ANN)


#    1) 패키지 로딩
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers.core import Dense


#    2) 데이터 가공
#       스케일링
people_x = people_x / 255

#       NN 학습을 위한 Y 변환
people_y_dummy = pd.get_dummies(people_y).values

#       데이터 분리
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y, train_y_dm, test_y_dm = train_test_split(people_x, 
                                                                           people_y, 
                                                                           people_y_dummy, 
                                                                           random_state=0)


#    3) 모델링
#       - stopping rule 생성
from keras.callbacks import EarlyStopping
m_stop = EarlyStopping(monitor='val_loss', patient=5)


#       - 모델 저장
import os 
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :
    os.mkdir(MODEL_DIR)

modelpath = './model/people-{epch:03d}-{val_loss:.4f}.hdf5'  
from keras.callbacks import ModelCheckpoint  


checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', savel_bes_only=True) 


#      - 모델링
model = Sequential()
model.add(Dense(1500, input_dim = train_x.shape[1], activation = 'relu'))
model.add(Dense(800, activation = 'relu'))
model.add(Dense(400, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(train_y_dm.shape[1], activation = 'softmax'))         

model.compile(optimizer = 'sgd', 
              loss = 'crossentropy',
              metrics = 'accuracy')


# m_stop, checkpointer에서는 validation dataset으로 평가
# fit 당시에 학습 데이터(train)만 제공할 경우는 반드시 validation_split 지정
# fit 당시에 학습 데이터와 검증 데이터를 각각 전달 가능 > validation_data 옵션으로 전달
model.fit(train_x, train_y_dm, epochs=3000, batch_size = 10, 
          callbacks=[m_stop, checkpointer], 
          validation_split=0.2, 
          verbose = 0)        # 학습 과정에 대한 출력 여부(default : 1)

model.evaluate(test_x, test_y_dm)[1]


