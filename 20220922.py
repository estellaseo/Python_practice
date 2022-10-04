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
# pip install sklearn_SOM
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
# 1. 총분산 (total_ss)
#    - 군집과 상관없이 항상 고정
#    - (각 관측치 값 - 전체평균)의 제곱의 총 합
#    - 군집간 분산 + 군집내 분산

# 2. 군집내 분산 (within_ss)
#    - 각 관측치가 포함된 군집의 중심(평균)으로부터 떨어진 정도
#    - 군집분석이 잘 될수록 줄어듦(군집 수가 많아져도 작아짐)
#    - (1번 군집의 각 관측치 - 1번 군집의 평균)의 제곱의 총 합 

# 3. 군집간 분산
#    - 군집의 중심(평균)이 전체중심(전체평균)과 떨어진 정도
#    - 군집분석이 잘 될수록 커짐
#    - (각 군집의 중심 - 전체중심)의 제곱의 총 합

# 1) 군집 수의 결정 : elbow point 결정
# 2) 군집 모델 비교 : 군집간변동/총변동, 군집간변동/군집내변동(클수록 좋은 모델)


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

iris_x.mean()
iris_mean = iris_x.mean(axis = 0).reshape(1, -1)
np.sum((iris_x - iris_mean)**2)







