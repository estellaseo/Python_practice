# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# Impurity 불순도
# =============================================================================
# 1. 정의  
#    - 데이터(target data의 class)가 혼합되어 있는 정도
#    - 분류모델의 변수 중요도 측정 기준



# 2. 종류
# 1) 지니지수
#    - 분류 트리일 때 사용하는 불순도 측정 척도
#    - 한쪽 노드에서의 불순도를 측정(한 노드가 얼마나 혼합되어 있는지)
#    - 이진 분류(class가 2개)일 경우
#      > 한 class의 확률 : 1/2 > 1 - (1/2)**2 - (1/2)**2 = 1/2  최대
#      > 한 class의 확률 : 0 > 1 - 1**2 - 0**2 = 0              최소
#      > 한 class의 확률 : 1 > 1 - 1**2 - 0**2 = 0              최소

# 예제) 한 쪽 노드가 A그룹 5명, B그룹 3명, C그룹 2명으로 혼합되어 있을 때 지니지수값?
1 - (5/10)**2 - (3/10)**2 - (2/10)**2 = 0.62


# 2) 엔트로피지수
#    - 분류 트리일 때 사용하는 불순도 측정 척도
#    - 한쪽 노드에서의 불순도를 측정(한 노드가 얼마나 혼합되어 있는지)

# 예제) 한 쪽 노드가 A그룹 5명, B그룹 3명, C그룹 2명으로 혼합되어 있을 때 엔트로피지수?
-((5/10)*np.log2(5/10) + (3/10)*np.log2(3/10) + (2/10)*np.log2(2/10)) = 1.485


# 3) 정보소득(IG)
#    - 상위노드의 불순도(엔트로피지수)와 하위노드의 불순도의 차이를 나타내는 척도
#    - 정보소득이 클수록 상위노드와 하위노드의 차이가 큼 > 불순도 감도의 효과가 큼
#    - 정보소득이 큰 변수를 주로 선택 권장


# 4) 카이제곱통계량
#    - 분류 트리일 때 사용하는 불순도 측정 척도
#    - 상위노드가 갖는 혼합비율을 여전히 하위노드가 갖는지 측정
#    - 상위노드의 비율을 하위노드가 그대로 유지
#      > 기대도수와 실제도수의 차이가 발생하지 않음
#      > 카이제곱통계량은 작아짐
#      > 좋은 기준 X
#    - 카이제곱통계량이 클수록 상위노드와 하위노드가 다르다는 것을 의미함

# 예시) 성별에 따른 구매/비구매 분류에 대한 카이제곱통계량 측정 
#      (상위노드 구매 5명, 비구매 3명)

# 성별이 남자인경우 - 구매 3명, 비구매 1명
#      총 4명 중 원래 비율을 유지한다면 구매는 4 * 5/8 2.5명이 기대됨
#      총 4명 중 원래 비율을 유지한다면 비구매는 4 * 3/8 1.5명이 기대됨
#      따라서 카이제곱통계량은 (2.5 - 3)**2/2.5 + (1.5 -3)**2/1.5 = 1.6

# 성별이 여자인경우 - 구매 2명, 비구매 2명
#      총 4명 중 원래 비율을 유지한다면 구매는 4 * 5/8 2.5명이 기대됨
#      총 4명 중 원래 비율을 유지한다면 비구매는 4 * 3/8 1.5명이 기대됨
#      따라서 카이제곱통계량은 (2.5 - 2)**2/2.5 + (1.5 -2)**2/1.5 = 0.267



# 3. 특징
#    - 불순도는 상위노드일수록 큼
#    - 불순도의 최대 : 뿌리노드
#    - 불순도의 감소량을 최대로 하는 규칙이 best




# =============================================================================
# Decision Tree Regression
# =============================================================================
# - 의사결정나무를 이용한 회귀분석 가능
# - F 통게량, 분산감소량을 사용하여 최적의 분리조건을 찾아냄




# =============================================================================
# Random Forest
# =============================================================================
# - 앙상블 모형의 일종(여러 모델이 혼합되어 있는 형태의 분석 알고리즘)
# - 분류 분석(다수결), 회귀 분석(평균) 가능

# - 목적 : 서로 다른 트리 구성
#   1) 부트스트랩(Bootstrap) 
#      train dataset과 크기가 같은 복원추출을 허용한 서로 다른 dataset 구성
#   2) 임의성 정도
#      각 분류기준을 만들 때 모든 설명변수가 아닌 일부 설명변수를 추출한 후 이 중
#      변수중요도가 높은 변수를 사용(mtry, max_features)
#   3) 배깅(Bootstrap Aggregation)
#      각 트리의 결론을 최종 결합하여 결론을 내리는 방식



# 예시) Iris 데이터셋의 꽃의 분류(RF)
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']

from sklearn.model_selection import train_test_split
iris_x_tr, iris_x_te, iris_y_tr, iris_y_te = train_test_split(iris_x, iris_y, random_state = 0)


# 모듈 호출
import sklearn.tree                         # 의사결정나무
import sklearn.ensemble                     # 앙상블모형나무(RF, 에이다부스트, ...)

from sklearn.tree import DecisionTreeClassifier as dt_c
from sklearn.ensemble import RandomForestClassifier as rf_c
from sklearn.ensemble import RandomForestRegressor as rf_r


# 모델링 및 펵
m_rf = rf_c()
m_rf.fit(iris_x_tr, iris_y_tr)

dir(m_rf)                                   # 모델이 가지는 속성 목록
m_rf.feature_importances_                   # 변수 중요도

m_rf.score(iris_x_tr, iris_y_tr)            # 100(train dataset)
m_rf.score(iris_x_te, iris_y_te)            # 97.37(test dataset)


# 튜닝
m_rf?
# - n_estimators(default = 100) : 트리의 수
# - max_depth : 총 트리의 길이
# - min_samples_split(default = 2) : 최소 가지치기 기준
# - max_features(default = "auto", sqrt(n_features)) : 설명변수의 후보 수(임의성 정도)
# - bootstrap(default = True) : 복원추출을 허용한 같은 크기의 다른 dataset 구성 여부
# - n_jobs : 병렬 작업을 사용할 CPU core 수
# - random_state : seed 값 고정

score_tr = []; score_te = []
for i in range(2, 11) :
    m_rf = rf_c(min_samples_split = i, random_state = 0)
    m_rf.fit(iris_x_tr, iris_y_tr)
    score_tr.append(m_rf.score(iris_x_tr, iris_y_tr))
    score_te.append(m_rf.score(iris_x_te, iris_y_te))
    

# 매개변수 변화에 따른 예측률 변화 시각화
# 참고 : spyder 그래프 출력을 위한 옵션 조절
# tools > Preference > IPython console > Graphics > Graphics backend
import matplotlib.pyplot as plt

plt.plot(np.arange(2, 11), score_tr, c = 'r', label = 'Train Score')
plt.plot(np.arange(2, 11), score_te, c = 'b', label = 'Test Score')
plt.legend()




# [ 연습 문제 ] cancer data 분류(DT, RF)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer_x = cancer['data']
cancer_y = cancer['target']

# split data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(cancer_x, cancer_y)

train_x.shape                            # (426, 30)
test_x.shape                             # (143, 30)


# 1) DT로 훈련 시 예측 점수

m_dt = dt_c()                            # 빈 모델 (매개변수 정의)
m_dt.fit(train_x, train_y)               # 학습

m_dt.score(train_x, train_y)             # 1
m_dt.score(test_x, test_y)               # 0.9020979020979021

m_dt.feature_importances_

# 변수 중요도 순서대로 정렬
s1 = Series(m_dt.feature_importances_, index = cancer['feature_names'])
s1.sort_values(ascending = False)

# 예측
m_dt.predict(test_x)                     # 모델에 의한 예측값
m_dt.predict_proba(test_x)               # 각 클래스 분류 확률


# 2) RF로 훈련 시 예측 점수
m_rf = rf_c()
m_rf.fit(train_x, train_y)

m_rf.score(train_x, train_y)             # 1
m_rf.score(test_x, test_y)               # 0.965034965034965

# 예측
m_rf.predict(test_x)
m_rf.predict_proba(test_x)


# 3) RF로 훈련 시 max_features에 대한 튜닝
score_tr = []; score_te = []
for i in range(1, (len(cancer['feature_names'])+1)) :
    m_rf = rf_c(max_features = i, random_state = 0)
    m_rf.fit(train_x, train_y)
    score_tr.append(m_rf.score(train_x, train_y))
    score_te.append(m_rf.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(score_tr, c = 'r', label = 'train score')
plt.plot(score_te, c = 'b', label = 'train score')
plt.legend()





