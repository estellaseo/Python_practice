# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 거리기반 모델
# =============================================================================
# - 데이터끼리의 거리로 유사성을 판단
# - 지도학습 > knn
# - 비지도학습 > 군집분석(k-means)


# [ 거리 기반 모델의 장단점 ]
# 장점
# - 계산이 간단

# 단점함
# - 이상치, 결측치에 매우 민감
# - 선택된 변수에 민감(중요도가 낮은 변수가 삽입될수록 거리 계산의 정확성이 떨어짐)
# - 스케일링 민감
# - 고차원에 부적합(속도, 각 변수의 영향력이 너무 낮게 해석)
# - 연속형 설명변수로 구성될 때 가장 효과적(이산형과 범주형일 때, 특히 범주형이 많을수록 부적합)



#[ 거리 측정 방식 ]
# 1. 연속형 변수 거리
#    - 유클리드 거리 : 가장 일반적인 거리 형태
#    - 맨하튼 거리 : 두 점의 절대값을 합한 값으로 측정
#    - 민코프스키 거리 : m차원 공간에서의 거리(m = 1 맨하튼거리, m = 2 유클리드거리)

#    - 표준화 거리 : 변수의 측정단위를 표준화한 거리
#    - 마할라노비스 거리 : 변수의 표준화와 함께 변수 간의 상관성을 고려한 거리. 
#                    특정 분포내에서 평균으로부터 떨어진 거리이므로 이상치 검정에도 사용

# 2. 명목형 변수 거리
#    - 단순일치계수(자카드계수) : 전체 변수 중 값이 일치하는 변수의 비율
#                             (0~1의 값을 가지며 1에 가까울수록 유사성이 강하다)
#    - 자카드거리 : 1 - 자카드계수
#                 (0~1의 값을 가지며 1에 가까울수록 유사성이 약하다)
#    - 순서형거리 : 순서형 범주형을 갖는 경우(예를들면 상,중,하와 같은) 순서(숫자)로 
#                 변경한 후 순서의 차이를 사용하여 측정한 거리



# [ 최대 우도 추정법(MLE, Maximum Likelihood Estimation) ]
# 어떤 모수가 주어졌을 때, 원하는 값들이 나올 가능도를 최대로 만드는 모수를 선택하는 방법
# (모수의 점 추정방식)

# 우도(likelihood) = 가능성
# 어떤 시행의 결과가 주어졌다고 할 때, 주어진 가설이 참일 때 그 결과가 나오는 정도



# [ EM(Expectation-Maximization) ]
# 관측되지 않은 잠재변수에 의존하는 확률 모델에서 최대 우도나 최대 사후 확률을 갖는 모수의 
# 추정값을 찾는 반복적인 알고리즘



# 예시) k-means 실습
# 1. iris data 불러오기
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']


# 2. kmeans를 사용한 군집분석 수행
from sklearn.cluster import KMeans
m_kmeans = KMeans(3)
iris_predict = m_kmeans.fit_predict(iris_x)


# 3. 산점도(원본, 군집분석결과)
#    1) mglearn
import mglearn
# 원본
mglearn.discrete_scatter(iris_x[:, 2], iris_x[:, 3], iris_y)
# 군집분석결과
mglearn.discrete_scatter(iris_x[:, 2], iris_x[:, 3], iris_predict)

#    2) matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
#  원본
ax[0].scatter(iris_x[:, 2], iris_x[:, 3], c = iris_y)
ax[0].set_title('Iris raw data')
# 군집분석 결과
ax[1].scatter(iris_x[:, 2], iris_x[:, 3], c = iris_predict)
ax[1].set_title('Iris predicted data')



# [ SOM(Self Organizing Map, 자기 조직화 지도) ]
# - 인공신경망 기반 군집 분석
# - 고차원의 벡터를 시각화할 수 있는 2차원의 저차원으로
# - 자율 학습 방법에 의한 클러스터링 방법 적용
# - 입력 변수의 위치 관계 그대로 보존
# - 클러스터 수 지정 X


# 예시) SOM 실습
# 1. iris data 불러오기
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']


# 2. SOM 군집분석 수행
pip install sklearn_SOM
from sklearn_som.som import SOM

SOM(m = 3,                           # 격차크기(x)
    n = 3,                           # 격차크기(y)
    dim = 3,                         # input dimension(설명변수의 수)
    lr = 1,                          # 학습률(가중치 수정 정도)
    max_iter = 3000)                 # 최대반복 수

m_som2 = SOM(3, 3, 4)
iris_predict2 = m_som2.fit_predict(iris_x)

m_som3 = SOM(3, 3, 4, lr = 0.1)
iris_predict3 = m_som3.fit_predict(iris_x)

iris_x_2 = iris_x[:, 2:4]
m_som4 = SOM(2, 2, 2, lr = 0.1)
pred = m_som4.fit_predict(iris_x_2)


# 3. 시각화(학습률 별 클러스터링 변화)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4)

ax[0].scatter(iris_x[:, 2], iris_x[:, 3], c = iris_y)
ax[0].set_title('Iris raw data')

ax[1].scatter(iris_x[:, 2], iris_x[:, 3], c = iris_predict2)
ax[1].set_title('SOM predicted data (lr = 1)')

ax[2].scatter(iris_x[:, 2], iris_x[:, 3], c = iris_predict3)
ax[2].set_title('SOM predicted data (lr = 0.1)')

ax[3].scatter(iris_x_2[:, 0], iris_x_2[:, 1], c = pred)
ax[3].set_title('SOM 2 variables data')



# [군집분석 평가]
# 1. 총분산 : 군집과 상관없이 고정
#            (각 관측치 값 - 전체평균)의 제곱의 총 합

# 2. 군집내 분산 : 군집분석이 잘 될수록 줄어듦(군집 수가 많아져도 작아짐)
#                (1번 군집의 각 관측치 - 1번 군집의 평균)의 제곱의 총 합 
# 3. 군집간 분산 : 군집분석이 잘 될수록 커짐
#                (각 군집의 중심 - 전체중심)의 제곱의 총 합



# 군집분석 평가 함수
def f_cluster_score(x_data, pred) :
    within_ss = []
    for i in Series(pred).unique() :
        vin = x_data[pred == i] - x_data[pred == i].mean(axis = 0)
        within_ss.append((vin**2).sum())
    
    between_ss = []
    for i in Series(pred).unique() :
        vout = x_data[pred == i].mean(axis = 0) - x_data.mean(axis = 0)
        between_ss.append((vout**2).sum())
    between_ss2 = sum(between_ss)
        
    for i in range(1, len(Series(pred).unique()) + 1) :
        print('군집내 분산 (군집 %s) ' %i, within_ss[i-1])
    print('군집내 분석 총합 %s' %sum(within_ss))
    print('군집간 분산 %s' %between_ss2)
    
    
f_cluster_score(iris_x, pred)
f_cluster_score(iris_x, iris_predict)

 


# =============================================================================
# 회귀분석
# =============================================================================
# - 지도학습 > Y 연속형일 경우 사용
# - 하나 이상의 독립변수들이 종속 변수에 미치는 영향을 추정할 수 있는 통계기법
# - 통계적 회귀분석 모델 : Regression
# - 비통계적 회귀분석 모델 : Decision Tree, Random Forest, SVM, knn, ANN, etc.


# [ 회귀분석(Regression)의 특징 ]
# - 이상치 민감
# - 변수 조합에 민감
# - 통계적 가정 하에 만들어진 모델


# [ 상관관계와 인과관계 ]
#   1) 아이스크림 판매량, 기온 > 상관관계, 인과관계 인정
#   2) 익사 수, 아이스크림 판매량 > 상관관계 인정, 인과관계 인정X


# [ 회귀분석 모형의 가정 ]
# - 선형성 : 독립변수의 변화에 따라 종속변수도 일정 크기로 변화되는지 확인(산점도로 판단)
# - 독립성 : 잔차와 독립변수가 서로 독립적인지 확인(잔차 산점도, 더빗-왓슨 검정)
# - 등분산성 : 잔차들의 분산이 일정한지 확인(잔차 산점도로 진단)
# - 정규성(정상성) : 잔차항이 정규 분포를 따르는지 확인 - 매우 중요함
#                 (히스토그램, Q-Q plot, 샤피로-윌크 검정, 콜모고로프-스미르노프 적합성 검정)

# - 적합도 검정 > 정규성 검정
#             > 기대분포를 따르는지 검정(카이제곱 검정)


# [ 참고 ]
# normal Q-Q : 정규성에서 벗어나는 이상치 확인 가능
# 등분산성 위배 > y가 커질수록 잔차가 커짐
# 통계적 분석 : 시계열, 회귀 등 통계적 모델에 대해서 R 결과가 Python보다 더 정확할 수 있음


# [ 회귀분석 모형 ]
# - 단순(선형)회귀 : 독립변수가 1개이며 종속변수와의 관계가 선형인 경우
# - 다중회귀 : 독립변수가 k개이며 종속변수와의 관계가 선형(1차 함수)
# - 다항회귀 : 독립변수와 종속변수와의 관계가 1차 함수 이상인 관계
# - 로지스틱회귀 : 종속변수가 범주형인 경우의 회귀분석
#               (단순, 다중 로지스틱 회귀, 이항, 다항 로지스틱 회귀)
# - 곡선회귀 : 독립변수가 1개이며 종속변수와의 관계가 곡선인 경우
# - 비선형회귀 : 회귀식 모양이 미지의 모수들과 선형관계가 아닌 경우


# [ 회귀 모형의 유의성 검정 ]
# 총편차 : y - ȳ
# 총변동량(SST) = 오차제곱합(SSE) + 회귀제곱합(SSR)
#    y - ȳ     =    (y - ŷ)    +   (ŷ - ȳ)


# 회귀 계수 추정 - MSE (평균제곱오차)
# - 회귀 분석 모형의 오차항은 서로 독립이고 평균이 0, 분산이 σ²인 정규분포를 따르는 것으로 가정
# - 오차의 크기를 측정하는 용도로도 사용(신경망 모델의 손실 함수)

# F분포 상에서의 가설검정을 했으므로 p-value 값이 0.05보다 작을 경우 모형이 유의하다고 봄(H1)


# 모든 회귀 계수의 유의성 검정이 통계적으로 검증되어야 선택된 변수들의 조합을 최종 모형으로 사용
# 모형이 유의하더라도 회귀 계수는 유의하지 않을 수 있음
# 모든 회귀 계수가 유의하더라도 모형이 유의하지 않을 수 있음(설명력이 낮은 모델)

# R² = SSR(회귀제곱합) / SST(총변동량)
# 결정계수는 모형에 유의하지 않은 변수가 증가해도 증가함(주의점)
# 보완 > 수정된 결정계수(Adjusted R-Squared)
#       : 수정된 결정계수는 설명변수 개수가 다른 모델간 비교할 때 수치로 사용됨



# [ 다중선형회귀분석 검정 ]
# 1) 모형 유의성 검정 : F 통계량으로 검정
# 2) 회귀 계수 유의성 검정 : t 통계량으로 검정
# 3) 모형의 설명력 : 결정계수와 수정된 결정계수를 사용
# 4) 모형의 적합성 : 모형이 데이터를 잘 적합하고 있는지 확인. 잔차와 종속변수의 산점도로 확인
# 5) 다중공선성 : 설명변수 사이의 선형관계 여부 확인. 분산팽창요인, 상태지수 등으로 측정
#                > 다중공선성이 있는 경우 회귀계수를 신뢰하기 어려움



# [ 다중공선성 문제 해결 방법 ]
# 다중공선성이 높다고 판단할 수 있을 때
# 변수간 상관계수 확인 corr(train_x)
# 상관계수가 여러 변수에 걸쳐서 높을 때 가장 문제가 되는 변수가 될 가능성이 높음

# 1) 문제 있는 변수 제거
# 2) 주성분 분석을 이용한 차원 축소 > 변수간 인과관계 파악 불가
# 3) 릿지, 라쏘 등의 모형 대체



# [ 분산팽창지수 ]
# VIF가 4보다 크면 다중공선성 존재. 10보다 큰 경우 심각한 문제가 있는 것으로 해석
# VIF = 1 / 1-R²
# 변수별로 각각(x1, x2, x3) y로 두고 결정계수를 구함 > VIF 함수 이용



# [ 변수 선택 방법 ]
# 1) 전진 선택법(Forward Selection)
#    : 절편만 있는 상수 모형으로부터 시작하여 중요한 변수들을 추가하는 방식
# 2) 후진 제거법(Backward Elimination)
#    : 전체 변수가 모두 삽입된 모형으로부터 가장 중요하지 않은 변수들을 제거하는 방식
# 3) 단계적 방법(Stepwise Method)
#    : 전체 변수가 모두 삽입된 모형으로부터 가장 중요하지 않은 변수들을 제거하면서
#      제거된 변수들 중 다시 추가하는 경우를 고려하는 방식

# AIC(아카이케 정보 기준 : Akaikie Information Criteria)





