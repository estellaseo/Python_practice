# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 데이터 전처리
# =============================================================================
# 데이터의 품질을 높이고 분석의 정확도를 위해 필요한 데이터 정제 작업
# data cleansing / data cleaning
# 데이터 정제 > 결측값 처리 > 이상값 처리 > 분석 변수 처리


# [ 데이터 정제 ]
#  - 중복값 제거
#  - 결측치 처리
#  - 이상치 처리



# [ 데이터 정제 절차 ]
# 1. 오류 원인 분석
#    1) 결측값
#       - 필수적인 데이터가 입력되지 않고 누락된 값
#       - 해결방법 : 제거, 통곗값 혹은 분포 기반 대체

#    2) 노이즈
#       - 실제는 입력되지 않았지만 입력되었다고 잘못 판단된 값
#       - 해결방법 : 제거, 평균이나 중간값 대체

#    3) 이상값
#       - 데이터 범위에서 많이 벗어난 값
#       - 해결방법 : 제거, 최대/최소값 대체

# 2. 정제 대상 선정
#    - 모든 데이터 대상
#    - 특별히 데이터 품질 저하의 위협이 있는 데이터에 대해 더 많은 정제 활동 수행
#    - 외부 데이터일수록, 비정형 데이터일수록 품질 저하 가능성 높음

# 3. 정제 방법 결정
#    1) 삭제
#       - 부분 혹은 전체 삭제 처리
#    2) 대체
#       - 평균값, 최빈값, 중앙값 등으로 대체. 왜곡 발생 위험 있음
#    3) 예측값 삽입
#       - 회귀식 등을 이용한 예측값 생성 후 삽입(ex. 시계열 데이터 보간 작업 등)

# 4. 데이터 일관성을 위한 정제
#    1) 변환
#       - 다양한 형태의 표현된 값을 하나의 일관된 형태로 변환(성별, 날짜 형식 등)
#    2) 파싱
#       - 데이터 정제 규칙을 적용하기 위한 유의미한 최소 단위로 분할하는 작업
#       - 주민등록번호에서 생년월일, 성별로 분한
#    3) 보강
#       - 변환, 파싱, 수정, 표준화 등을 통한 추가 정보 반영
#       - 주민등록번호를 통해 성별 추출 후 추가 보강


# [ 결측치 처리 ]
# 1. 결측값 종류
#    1) 완전 무작위 결측(Missing Completely at Random)
#    2) 무작위 결측(Missing at Random)
#    3) 비 무작위 결측(Missing not at Random)

# 2. 결측값 부호화
#    - 결측값으로 표현 가능한 부호로 대체
#    - NA, NaN, inf, NULL

# 3. 결측값 처리 방법
#    1) 단순 대치법
#       - 완전 분석법(단순 삭제)
#         : 결측값이 특정 비율을 초과하는 경우 컬럼 삭제. 데이터 손실 발생
#       - 평균 대치법
#         : 추정량의 표준오차가 과소 추정되는 문제 발생. 중앙값, 최빈값으로도 대치
#       - 단순 확률 대치법
#         : Hot-Deck 현재 진행 중인 연구에서 비슷한 성향을 가진 응답자의 자료로 대체
#         : Cold-Deck 외부 출처에서 비슷한 성향을 가진 응답자의 자료로 대체
#         : Nearest Neighbor KNN모델을 참조하여 대체
#    2) 다중 대치법
#       - 단순 대치법을 m번 대치를 통해 m개의 가상적 완전한 자료를 만들어 분석하는 방법
#       - 대치 > 분석 > 결합




# [ 분석 변수 처리 ]
#  - 변수 변환
#  - 파생 변수 생성 : 기존의 데이터를 활용하여 기존과 다른 데이터 창출
#  - 차원 축소




# =============================================================================
# 차원 축소
# =============================================================================
# - 차원, (설명/독립)변수, 특성(feature) > 독립변수
# - 많은 설명변수(고차원) > 보다 적은 설명변수(저차원)
# - 고차원 데이터의 학습은 연산비용 및 과대적합의 위험성이 높음
# - 가장 간단한 방법 : 변수 삭제 but 데이터 손실 발생
# - 기존 변수를 모두 사용하여 기존 변수의 가중합(선형결합)의 결과로 새로운 인공변수 유도
# - EDA 시 시각화 목적으로 많이 사용(2, 3차원 선상)
# - 데이터 손실 없이 모든 데이터를 시각하기 위함
# - 비지도 학습 기법


# 1. 주성분분석(Principal Component Analysis : PCA)
#    - 회귀분석에서 자주 이용되는 차원축소 기법
#    - 기존 데이터가 갖는 분산을 최대한 유지하는 방식으로 변수 결합(가중치 추정)
#    - 공분산 행렬, 상관행렬을 이용한 방식의 가중치 추정
#    - 유도된 첫번째 인공변수가 가장 많은 분산 설명력을 가짐
#    - 두번째 인공변수는 첫번째 인공변수가 설명하지 않는 나머지 분산 설명력을 가짐
#    - 유도된 인공변수끼리 서로 독립적 관계
#    - 총 분산 설명력이 80% 이상이 되는 인공변수 수 결정



# 예시) iris data 차원 축소 - 2차원 공간 시각화
# 1. 데이터 불러오기
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']


# 2. 스케일링
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
iris_x_sc = m_sc.fit_transform(iris_x)


# 3. PCA 적용
from sklearn.decomposition import PCA
m_pca = PCA(2)
m_pca.fit(iris_x_sc)                    # 가중치 추정
m_pca.components_                       # 추정된 가중치

C1 = 0.52106591*X1 -0.26934744*X2 + ...
X1 = iris_x_sc[:, 0]..

m_pca.transform(iris_x_sc)              # 추정된 가중치를 이용한 변수 변환
iris_x_sc_pca2 = m_pca.fit_transform(iris_x_sc) # fit과 transform 동시에 진행

m_pca.explained_variance_               # 분산 성명 정도
m_pca.explained_variance_ratio_         # 분산 설명력(비율)
m_pca.explained_variance_ratio_.sum()   # 총 분산 설명력(95.81%)


# 4. 시각화
import mglearn
mglearn.discrete_scatter(iris_x_sc_pca2[:, 0],
                         iris_x_sc_pca2[:, 1],
                         iris_y)


# SVM 모델 적용
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
iris_x_tr, iris_x_te, iris_y_tr, iris_y_te = train_test_split(iris_x_sc_pca2, 
                                                              iris_y,
                                                              random_state = 0)
m_svm = SVC()
m_svm.fit(iris_x_tr, iris_y_tr)
m_svm.score(iris_x_te, iris_y_te)       # 0.9210526315789473




# [ 연습 문제 ]
# iris data를 2개 클래스를 갖는 데이터로 변경 후(setosa 제외)
# PCA + SVM 모델의 2차원 결정경계 시각화
# 1. 데이터 불러오기
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']

iris_x = iris_x[iris_y != 0, :]
iris_y = iris_y[iris_y != 0]


# 2. 스케일링
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
iris_x_sc = m_sc.fit_transform(iris_x)


# 3. PCA 적용
from sklearn.decomposition import PCA
m_pca = PCA(2)
iris_x_sc_pca2 = m_pca.fit_transform(iris_x_sc)


# 4. SVM 모델 적용 및 시각화
from sklearn.svm import SVC
import mglearn

m_svm = SVC()
m_svm.fit(iris_x_sc_pca2, iris_y)

mglearn.discrete_scatter(iris_x_sc_pca2[:, 0], iris_x_sc_pca2[:, 1], iris_y)
mglearn.plots.plot_2d_separator(m_svm, iris_x_sc_pca2) 


