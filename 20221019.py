# -*- coding: utf-8 -*-

# =============================================================================
# ANN regression
# =============================================================================

# 예제) Boston 주택 가격 예측(회귀)
# 1. 데이터 로딩
from sklearn.datasets import load_boston
boston = load_boston()

boston_x = boston['data']
boston_y = boston['target']


# 2. 데이터 가공
# X 데이터 스케일링
from sklearn.preprocessing import StandardScaler as standard

m_sc = standard()
boston_x_sc = m_sc.fit_transform(boston_x)


# 3. train/test dataset split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(boston_x_sc, boston_y, random_state = 0)


# 4. 모델정의
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(52, input_dim = train_x.shape[1], activation = 'relu'))
model.add(Dense(26, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(1))


# 5.컴파일
# R**2 값 계산 함수
def r_square(y_true, y_pred):
    from keras import backend as K
    SSE =  K.sum(K.square(y_true - y_pred))          # (y - yhat)^2
    SST = K.sum(K.square(y_true - K.mean(y_true)))   # (y - ybar)^2
    return (1 - SSE/(SST + K.epsilon()))             # K.epsilon : 오차항(분모 0인 상황 방지)


model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',
              metrics = r_square)


# 6. 정지규칙과 모델 저장
# 정지 규칙 생성
from keras.callbacks import EarlyStopping
m_stop = EarlyStopping(monitor='val_loss', patience=5)   

# 모델 저장
import os
modelpath = './model/boston-ann-{epoch:03d}-{val_loss:.4f}.hdf5'
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)


# 7. fitting
model.fit(train_x, train_y, epochs = 2000, batch_size = 10, validation_split = 0.2,
          callbacks = [m_stop, checkpointer])



# 8. 평가
model.evaluate(test_x, test_y)[1]                # 0.7844



# 9. 다른 모델과 비교
#    1) Linear Regression
from sklearn.linear_model import LinearRegression
m_reg1 = LinearRegression()
m_reg1.fit(train_x, train_y)
m_reg1.score(test_x, test_y)                     # 0.6354


#    2) Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
m_reg2 = GradientBoostingRegressor()
m_reg2.fit(train_x, train_y)
m_reg2.score(test_x, test_y)                     # 0.8254




# =============================================================================
# ANN을 이용한 Boston 주택 가격 예측(interaction effect 고려)
# =============================================================================
# 1. 데이터 로딩
from mglearn.datasets import load_extended_boston
boston_x_ex, _ = load_extended_boston()          # _ 자리 데이터 > 불러오지 않음
boston_x_ex.shape                                # (506, 104)


# 2. 데이터 스케일링
# X 데이터 스케일링
from sklearn.preprocessing import StandardScaler as standard

m_sc = standard()
boston_x_ex_sc = m_sc.fit_transform(boston_x_ex)


# 3. train/test dataset split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(boston_x_ex_sc, boston_y, random_state = 0)


# 4. 모델정의
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(52, input_dim = train_x.shape[1], activation = 'relu'))
model.add(Dense(26, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(1))


# 5.컴파일
# R**2 값 계산 함수
def r_square(y_true, y_pred):
    from keras import backend as K
    SSE =  K.sum(K.square(y_true - y_pred))          # (y - yhat)^2
    SST = K.sum(K.square(y_true - K.mean(y_true)))   # (y - ybar)^2
    return (1 - SSE/(SST + K.epsilon()))             # K.epsilon : 오차항(분모 0인 상황 방지)


model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',
              metrics = r_square)


# 6. 정지규칙과 모델 저장
# 정지 규칙 생성
from keras.callbacks import EarlyStopping
m_stop = EarlyStopping(monitor='val_loss', patience=5)   

# 모델 저장
import os
modelpath = './model/boston-ann-{epoch:03d}-{val_loss:.4f}.hdf5'
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)


# 7. fitting
model.fit(train_x, train_y, epochs = 2000, batch_size = 10, validation_split = 0.2,
          callbacks = [m_stop, checkpointer])



# 8. 평가
model.evaluate(test_x, test_y)[1]                # 0.7844




# 
