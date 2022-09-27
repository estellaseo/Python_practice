# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 변수 변환과 파생 변수 생성
# =============================================================================
# 결측치, 이상치보다 예측률에 더 많은 영향을 미치기도 함
# 의미적 형태/분포적 형태 


# [ 변수 변환 ]
# - 분석을 위해 불필요한 변수를 제거하고 변수를 변환하여 새로운 변수를 생성시키는 과정
# - 수집된 변수 그대로가 아닌 다른 형태의 변수를 가공하는 기법


# [ 변수 변환 방법 ]
# 1. 단순 기능 변환
#    - 변수를 단순한 형태로 변환하는 기법
#    - 변수의 분포를 변경할 목적으로 주로 사용
#    - 로그 변환, 역수 변환, 루트 변환, 제곱 변환 등
#    - 로그 변환 : 분포가 왼쪽으로 기울어진 것을 감소
#                 0 또는 음수 값 적용 불가
#                 이산형 변수는 로그 변환을 미리 수행하기도 함
#    - 역수 변환 : ex) 총구매액/방문일수 
#                 변환 후 단위 변경 주의


# 2. 비닝(Binning)
#    - 연속 데이터 값을 몇 개의 bin으로 분할하는 방법
#    - 데이터로부터 잡음을 제거하는 데이터 평활화(equalization)에도 사용됨
#    - 분류 예측 모델보다는 회귀 예측 모델에 의미가 있음


# 3. 정규화(normalization) 및 표준화(standardization)
#    - 데이터를 특정 구간으로 바꾸는 척도법
#    - 변수 스케일링이라고도 함
#    - min-max scaling : 변수 값의 최솟값을 0, 최댓값을 1로 맞추는 작업
#    - standard scaling : 평균을 0, 분산을 1로 변경
#    - range의 차이가 동일한 것처럼 보여도 분산이 다를 수 있기 때문에 scaling 필수



# [ 파생 변수 ]
# - 기존 변수에 특정 조건 혹은 함수 등을 사용하여 새롭게 재정의한 변수를 의미
# - 변수 생성 시 논리적 타당성과 기준을 가지고 형성(but 주관적일 수밖에 없음)
#   1) 단위 변환
#      : 변수의 단위 혹은 척도를 변환하여 새로운 단위로 표현(ex. 총구매액/총구매건수)
#   2) 표현 형식 변환
#      : 표현되는 형식을 변환(ex. 날짜 데이터를 요일로 변환)
#   3) 요약 통계량 변환
#      : 요약 통계량 등을 활용(ex. 누적 방문 횟수, 평균 사용 금액) 
#   4) 변수결합
#      : 다양한 함수나 수학적 결합을 통해 새로운 변수 정의


# interaction effect(교호작용, 상호작용)
# 변수들끼리의 상호곱 형태로 의미성 존재 가능성
# 회귀에서 특히 


# 예시) iris dataset 분류(교호작용 의미 확인)
# 1. data 불러오기
from sklearn.datasets import load_iris
iris = load_iris()

iris_x = iris['data']
iris_y = iris['target']


# 2. data scaling
from sklearn.preprocessing import StandardScaler as standard

m_sc = standard()
iris_x_sc = m_sc.fit_transform(iris_x)


# 3. interaction
#    4개 변수 중 2차항의 상호작용 고려 
#    > X1X2, X1X3, X1X4, ... 총 6개의 조합 + 각 변수의 제곱합 + 상수항
from sklearn.preprocessing import PolynomialFeatures as poly

m_poly = poly(degree = 2)                      # default : 2(2차항)
m_poly.fit(iris_x_sc)                          # 발생 가능한 변수 조합 탐색(실제 학습 X)
iris_x_sc_poly = m_poly.transform(iris_x_sc)   # 교호작용 결과값 계산 및 리턴

iris_x_sc_poly.shape                           # 기존변수(4) + 상수항(1) + 제곱항(4) + 교호작용(6)

m_poly.get_feature_names()                     # 변환된 변수공의 결합 형태 제공
m_poly.get_feature_names(iris['feature_names'])# 기존 변수 이름 형태 제공

cols = Series(iris['feature_names']).str.replace(' (cm)', '', regex = False).str.replace(' ', '_')
poly_cols = m_poly.get_feature_names(cols)


# 4. 변수중요도 확인
from sklearn.tree import DecisionTreeClassifier as dt_c

m_dt = dt_c()
m_dt.fit(iris_x_sc_poly, iris_y)
s_var_imp = Series(m_dt.feature_importances_, index = poly_cols)

s_var_imp.sort_values(ascending = False)


# 5. 변수 선택
cols_sel = s_var_imp.sort_values(ascending = False)[:4].index
df_poly = DataFrame(iris_x_sc_poly, columns = poly_cols)
df_sel = df_poly.loc[:, cols_sel]


# 6. 선택된 변수에 대한 모델링
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_sel, iris_y, random_state = 0)

m_dt = dt_c()
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)                   # 1.0
m_dt.score(test_x, test_y)                     # 0.9736842105263158















