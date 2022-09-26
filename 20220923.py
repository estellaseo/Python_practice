# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 분석 모형 진단
# =============================================================================
# 일반화의 가능성, 모델이 가진 오류, 모델에 대한 가정 확인


# [ 분석 모델 평가 ]
# - 결정 계수(R-square), 수정된 결정계수(Adjusted R-square)
# - 정확도(Accuracy)
# - ROC Curve
# - AUC


# [ 분석 모형 진단 ]
# 1. 일반화 오류(과대 적합) 여부 확인
#    - 예시) train score - test score > 5 ~ 10
#    - Cross Validation을 통한 일반화된 점수 필요

# 2. 학습 오류(과소 적합) 여부 확인
#    - 예시) train score - test score < 3 ~ 10
#    - Cross Validation을 통한 일반화된 점수 필요

# 3. 기본 가정에 대한 진단

# 4. 결과 시각화
#    - 정보 전달과 설득의 목적
#    - 설명, 탐색, 표현이 있음




# =============================================================================
# 교차 검증
# =============================================================================
# 1. Hold-out 교차 검증
#    - 전체 dataset을 비복원추출 방법을 이용하여 random하게 학습 데이터와 평가 데이터로 나눔
#    - 일반적으로 5:5, 7:3, 8:2 등의 비율로 나눔
#    - 데이터를 어떻게 나누느냐에 따라 결과가 많이 달라질 수 있음
#    - train, test, validation dataset 세가지로 나누기도 함


# 2. 다중 교차 검증
#    1) 랜덤 서브 샘플링
#       - hold-out 교차 검증을 반복하여 데이터 손실 방지를 해결
#       - 각 샘플들의 학습 및 평가 횟수를 제한하지 않아 특정 데이터만 학습될 수 있음
#         (학습 및 평가 어디에도 사용되지 않은 데이터가 발생될 수 있음)
#       - 따라서 실제로 사용하기 보다는 이론적으로 존재하는 기법

#    2) K-fold Cross Validation
#       - 가장 일반적인 교차 검증 기법
#       - 데이터 집합을 무작위로 동일 크기를 갖는 K개 부분 집합으로 나누고, 
#         그 중 1개 집합을 평가 데이터, 나머지를 학습 데이터로 선정
#       - 모든 데이터를 빠짐 없이 학습과 평가에 사용
#       - K값이 증가할수록 수행 시간과 계산량 증가
#       - k = 3 일 때 train 66.66, test 33.33 / k = 4 일 때 train 75, test 25

#    3) Leave-One-Out Cross Validation(LOOCV)

#    4) Leave-P-Out Cross Validation(LpOCV)

#    5) RLT(Repeated Learning-Testing)
#       - 랜덤 비복원 추출 방식

#    6) 부트스트랩(Bootstrap)
#       - 단순 랜덤 복원추출 방법을 활용하여 동일한 크기의 표본을 여러 개 생성 및 검증
#       - 부트스트랩으로 N개 샘플 생성, 한 번도 선택되지 않은 약 36.8% 데이터를 평가에 사용




# =============================================================================
# 적합도 검정
# =============================================================================
# 표본 집단의 분포가 주어진 특정 이론(분포)을 따르고 있는지 검정
# 가정된 확률이 정해져 있는 경우(카이제곱 검정)와 정해져 있지 않은 경우(정규성 검정)로 구분

# 가정 확률이 정해진 경우의 검정
# - 카이제곱 적합도 검정
# - 주어진 데이터가 정해진 확률(기대도수)과 일치하는지 검정

# 가정된 확률이 정해지지 않은 경우의 검정
# - 정규성 검정
# - 샤피로-윌크 검정, K-S(콜모고로프-스미르노프) 검정


# [ 카이제곱 적합도 검정 ]
# - 관찰된 빈도가 기대되는 빈도(가정 확률)와 유의미하게 다른지 검정
# - 기대빈도와 관찰빈도의 차의 제곱을 기대빈도로 나눈 값이 자유도 k-1를 갖는 
#   카이제곱 분포를 따른다는 가정
# - H0 : 관찰빈도가 기대빈도와 일치함
#   H1 : 관찰빈도가 기대빈도와 일치하지 않음 > 오른쪽 검정

# - 기대도수 대비 관찰도수가 얼마나 다르게 유의한지 평가하는 지표
# - 기대균등 여부에 대한 가설 검정 수행


# [ 샤피로 윌크 검정 ]
# - 정규분포를 따르는지에 대한 검정 수행
# - shapiro.test()
# - 데이터 수가 3 ~ 5000개로 제한(그 이상일 경우 K-S 검정 사용)
# - H0 : 데이터가 정규분포를 따름
#   H1 : 데이터가 정규분포를 따르지 않음


# [ K-S 검정 ]
# - 데이터가 어떤 특정한 분포를 따르는가를 비교하는 검정 기법
# - 서로 다른 두 그룹이 동일한 분포인지도 검정 가능
# - ks.test(x, 'pnorm', mean = 0, sd = 1)
#   ks.test 수행 시 default 분포 : 표준정규분포
# - 비교 기준이 되는 분포를 정규 분포로 두어 정규성 검정 가능
# - H0: 두 그룹의 데이터는 동일한 분포를 따름
#   H1: 두 그룹의 데이터는 서로 다른 분포를 따름



# [ 모수 유의성 검정 ]

# 모평균   1, 2개   Z-검정(모분산을 알 경우)
#                  T-검정(모분산을 모를 경우)
#        3개 이상

# 모분산




# =============================================================================
# 로지스틱 회귀
# =============================================================================
# 반응변수가 범주형인 경우 적용되는 회귀분석 모형(분류 모델)
# 반응변수의 각 범주에 속할 확률이 얼마인지를 추정하여 추정 확률을 기준치에 따라 분류하는 모델
# 변수들의 선형 결합(설명변수들의 가중합 형태)으로 log(odds)를 추정하는 방식
# 성공확률을 최대로 하는 회귀 계수를 추정 > 최대우도추정법에 의한 추정


# [ 로지스틱 회귀 분석 종류 ]
# 1. 이항 로지스틱 회귀
#    : Y가 두 개의 범주로 구성
# 2. 다항 로지스틱 회귀
#    : Y가 세 개 이상의 범주로 구성


# [ 로지스틱 회귀식 ]
# 승산(odds) : 실패에 비해 성공할 확률의 비 p/(1-p)
log(odds) = b0 + b1X1                    # 단순선형회귀모델 - 설명변수 1개
log(P(Y=1) / P(Y=0)) = b0 + b1X1
P(Y=1)(1-P(Y=1)) = exp(bo + b1X1)



# [ 연습 문제 ] iris data에서 Sepal.Length 변수로 Setosa 여부 예측

# 1. data 불러오기
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']


# 2. 테스트를 위한 data 가공
iris_y[iris_y == 2] = 1
# or
iris_y = Series(iris_y).replace(2, 1)    # Setosa : 0, Setosa : 1

from sklearn.model_selectin import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, random_state = 0)


# 3. 모델링
# 1) 전체 설명변수로 예측률 확인
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)
m_lr.score(test_x, test_y)



# 2) 계수 확인
m_lr.coef_
m_lr.intercept_


# 3) test dataset의 첫 번째 관측치 결과 확인
test_x[0, :]


# 위 데이터의 각 변수값을 유도된 수식에 대입
log(odds) = 
P(Y = 1) = 
P


# 모델에 의한 예측
m_lr.predict(test_x[0, :].reshape(1, -1))
round(m_lr.predict_proba(test_x[0, :].reshape(1, -1)), 3)

m_lr.predict_proba(test_x[0, :].reshape(1, -1))[:, 1]  # 1의 확률 99.92%

p1 = m_lr.predict_proba(test_x)[:, 1]
p1.sort()


# 성공확률에 대한 시각화(S자모양 > 시그모이드 함수 시각화)
import matplotlib.pyplot as plt
plt.plot(p1)

p1 = m_lr.predict_proba(test_x)[:, 1]
np.where(p1 > 0.5, 1, 0)                  # P(Y=1)을 cutoff에 의해 해석된 예측값
m_lr.predict(test_x)                      # predict 함수가 위 기능을 대신함



