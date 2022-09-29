# -*- coding: utf-8 -*-

# =============================================================================
# 앙상블 분석
# =============================================================================
# 여러 가지 동일한 혹은 상이한 모형들의 결과를 종합하여 최종 의사결정에 활용하는 기법

# 앙상블 모형의 특징
# - 여러 개의 모델을 각각 훈련을 시킨 뒤 훈련 결과를 통합(평균 혹은 다수결)하여 최종 결론
# - 단일 모형으로 분석했을 때 보다 높은 신뢰성과 예측 정확도
# - 모형 해석이 어려움



# [Bagging : Bootstrap Aggregation, 배깅]
# - 다수의 부트스트랩 자료를 생성하고 각각 모델링한 후 결합하여 최종 예측 수행
# - 부트스트랩 : 주어진 자료에서 동일한 크기의 표본을 랜덤 복원 추출로 뽑은 자료
# - 평균 및 다수결을 통해 최종 예측 수행


# 배깅의 특징
# - 분산감소와 예측력 향상을 기대
# - 결측치에 민감하지 않음
# - 계산 복잡도가 다소 높음
# - 소량의 데이터일수록 유리
# - 목표변수, 입력변수의 크기가 작을수록 유리



# [ Boosting, 부스팅 ]
# - 잘못 분류된 개체들에 가중치를 적용, 새로운 분류 규칙을 만들고 이 과정을 반복하여
#   최종 예측 모형을 만드는 기법
# - 오분류 데이터에 높은 가중치를 부여하는 방식
# - 예측력이 약한 모형들을 결합하여 강한 예측 모형을 만드는 방식
# - 병렬효과 없음(트리가 서로 독립적이지 않기 때문에)
# - AdaBoost, GradientBoost 등


# 부스팅의 특징
# - 일반적으로 높은 성능(예측력)을 보이며 과대 적합도 없음
# - 대용량 데이터, 고차원 데이터 훈련에 유리
# - 비교적 높은 계산 복잡도


# 1. AdaBoost
#    - 오분류 데이터에 가중치를 높게 반영
#    - 보다 정교한 결정경계를 완성시키는 트리 생성

# 2. GradientBoost
#    - 오차를 줄이는 최적화 방식을 통해 트리 보완
#    - 오차 보완정도 > 학습률(learning rate) 
#    - 강력한 사전 가지치기 효과



# [ 랜덤 포레스트 ]
# 랜덤포레스트 주요 기법
# 1. 임의노드 최적화
#    : 훈련 목적함수를 최대로 만드는 노드 분할 함수 매개변수의 최적값을 구하는 과정

#    - 노드 분할 함수 : 트리와 노드마다 좌측, 우측의 자식노드로 분할하기 위해 사용되는 기준 함수
#    - 훈련 목적 함수 : 매개변수 최적값의 각 임계값들을 선택하는 기준
#    - 임의성 정도(p) : 각 트리들의 비상관화 수준 결정 요쇼. 클수록 상관성이 강함

# 2. 매개변수
#    - 트리의 수 : 클수록 훈련 및 예측시간은 증가하지만 정확도가 높아짐
#    - 최대 허용 깊이(max depth) : 너무 작으면 과소적합, 클수록 과대적합 위험 발생
#    - 임의성 정도 




# 예제) 분류과제 모델 비교
# 1. 데이터 로딩
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer_x = cancer['data']
cancer_y = cancer['target']

Series(cancer_y).value_counts()                # 357, 212


# 2. 데이터 분리



# 3. 모델링
from sklearn.model_selection import cross_val_score as cv

# 1) Decision Tree
from sklearn.tree import DecisionTreeClassifier as dt_c
m_dt = dt_c()
cv_score = cv(m_dt, cancer_x, cancer_y, cv=4)
cv_score.mean()                                  # 0.9174381956072097


# 2) Random Forest
from sklearn.ensemble import RandomForestClassifier as rf_c
m_rf = rf_c()
cv_score = cv(m_rf, cancer_x, cancer_y, cv=4)
cv_score.mean()                                  # 0.963126662070324


# 3) GBT
from sklearn.ensemble import GradientBoostingClassifier as gb_c
from sklearn.ensemble import AdaBoostClassifier as ada_c

m_gb = gb_c()
cv_score = cv(m_gb, cancer_x, cancer_y, cv=4)
cv_score.mean()                                  # 0.9613784103220723 

m_ada = ada_c()
cv_score = cv(m_ada, cancer_x, cancer_y, cv=4)
cv_score.mean()                                  # 0.9666601004629174 


# 4) XGB
# pip install xgboost
from xgboost.sklearn import XGBClassifier as xgb_c

m_xgb = xgb_c()
cv_score = cv(m_xgb, cancer_x, cancer_y, cv=4)
cv_score.mean()                                  # 0.9648872254506058 



