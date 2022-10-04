# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 회귀 모델
# =============================================================================
# 연속형 종속변수에 대한 예측 모델링
# 설명변수와 종속변수간의 인과관계 성립

# 1. 통계적 모델
#    - 이상치, 결측치 민감
#    - 통계적 가정(선형성, 독립성, 등분산성, 정규성) 성립되어야 모형을 신뢰할 수 있음
#    - 통계적 모델 > 통계적 평가 metrics 존재
#     (결정계수, 회귀계수 유의성 검정(t 분포 검정), 모형 유의성 검정(f 분포 검정))
#    - 선택된 변수에 매우 민감(사전 변수 선택 중요)

# 2. 비통계적 모델
#    - 통계적 가정 없이 예측 모델 생성
#    - 비통계적 모델 > 통계적 평가 metrics 존재 X 
#      : MSE, 결정계수 등으로 평가 가능(예측오차 기반 평가지표)




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




# 예제) 보스턴 주택 가격 데이터를 사용한 회귀 분석
# 1. 데이터 로딩
from sklearn.datasets import load_boston
boston = load_boston()

boston_x = boston['data']
boston_y = boston['target']

boston['feature_names']
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#        'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

print(boston['DESCR'])               # 변수 설명


# 2. 데이터 분리(train/test)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(boston_x, boston_y, random_state=0)


# 3. 모델링
#    3-1) sklearn
#         회귀계수와 모형의 유의성 검정 결과 리턴 X
from sklearn.linear_model import LinearRegression as reg
m_reg = reg()
m_reg.fit(train_x, train_y)

np.round(m_reg.coef_, 1)
m_reg.intercept_

m_reg.score(train_x, train_y)        # 0.7697 (결정계수 값)
m_reg.score(test_x, test_y)          # 0.6354


#    3-2) statmodels
#         회귀계수와 모형의 유의성 검정 결과 리턴
from statsmodels.formula.api import ols

# 데이터 합치기(데이터 프레임)
df_x = DataFrame(boston_x, columns = boston['feature_names'])
df_y = DataFrame(boston_y, columns = ['PRICE'])
df_boston = pd.concat([df_x, df_y], axis = 1)

# 원소를 결합하여 문자열 리턴(분리구분기호로 호출)
f1 = 'PRICE ~ ' + '+'.join(boston['feature_names'])
m_reg2 = ols(formula = f1, data = df_boston).fit()

# 회귀분석 결과 확인
print(m_reg2.summary())




# =============================================================================
# 비통계적 모델(회귀)
# =============================================================================
# 1) dt_r
from sklearn.tree import DecisionTreeRegressor as dt_r
m_dt = dt_r()
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)   # 100
m_dt.score(test_x, test_y)     # 63.81

# 2) rf_r
from sklearn.ensemble import RandomForestRegressor as rf_r
m_rf = rf_r()
m_rf.fit(train_x, train_y)
m_rf.score(train_x, train_y)   # 98.08
m_rf.score(test_x, test_y)     # 80.21

# 3) gb_r
from sklearn.ensemble import GradientBoostingRegressor as gb_r
m_gb = gb_r()
m_gb.fit(train_x, train_y)
m_gb.score(train_x, train_y)   # 98.3
m_gb.score(test_x, test_y)     # 82.76

# 4) knn_r
from sklearn.neighbors import KNeighborsRegressor as knn_r
m_knn = knn_r()
m_knn.fit(train_x, train_y)
m_knn.score(train_x, train_y)   # 70.66
m_knn.score(test_x, test_y)     # 46.16

# 변수 스케일링, 변수 선택 후 모델링 결과, k선택

# 5) SVR
from sklearn.svm import SVR
m_svr = SVR()
m_svr.fit(train_x, train_y)
m_svr.score(train_x, train_y)   # 24.4
m_svr.score(test_x, test_y)     # 8.37

# 변수 스케일링, 변수 선택 후 모델링 결과, C와 gamma 선택



# [ 회귀분석 전체 과정 ]
# 1. 모형 별 비교(sklearn, statmodels, 분류분석의 회귀모델)
# 2. 변수 상관성
#    - x, y의 상관성 : 선형성 가정
#                     상관성이 낮음 > 변수 제거는 위험(interaction effect)
#    - x끼리의 상관성 : 정보의 중복을 피하기 위하여 반드시 확인(다중공선성의 문제 발생 가능성)
# 3. 모형 학습 후
#    - 모형 가정 확인 : 잔차의 산점도(독립성, 등분산성)
#                     잔차의 히스토그램(정규성)
#    - 모형과 회귀계수 유의성 검정
# 4. 다중공선성 문제 진단
# 5. 다중공선성 해결(모형 변이 > 변수 선택, 파생 모델)
# 6. 예측력 문제 해결(변수 추가, 변수 변형 고려, interaction effect, binning)



# 예제) 회귀 분석
df1 = pd.read_csv('data/Interactions_Continuous.csv')

df_y = df1['Strength']
df_x = df1.drop('Strength', axis = 1) 


# 1. 변수간 상관관계(X 끼리)
df_x.corr()                   # 크게 상관관계가 높아보이지 않음
df1.corr()                    # 단독 변수의 형태로 Y의 영향을 미치는 변수 : Time
                              # 낮은 상관성을 보이는 temperature, pressure에 대한 고민
  
                              
# 2. 시각화
import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')
plt.rc('axes', unicode_minus = False)

import seaborn as sb
sb.heatmap(data = df_x.corr(),    # 데이터(2차원)
           annot = True,          # 주석(상관계수) 출력여부
           cmap = 'Reds')         # 색상



# 3. 모델링
#    1) sklearn
from sklearn.linear_model import LinearRegression as reg
m_reg = reg()
m_reg.fit(df_x, df_y)
m_reg.score(df_x, df_y)           # 0.8741

m_reg.coef_                       # array([ 0.26231425,  0.98490168, -4.46896022])
m_reg.intercept_                  # 144.1355243494118

# 추정된 회귀식
# Strength = 144.13 + 0.26*Temperature + 0.98*Pressure - 4.46*Time

# 잔차 산점도(x축: fitted value, y축: 잔차)
yhat = m_reg.predict(df_x)        # predict value, fitted value, 적합값, 예측값
res1 = df_y - m_reg.predict(df_x) # 잔차항

plt.scatter(yhat, res1)


#    2) statsmodels
# 데이터 입력 모델
from statsmodels.api import OLS
m_reg3 = OLS(df_y, df_x).fit()

print(m_reg3.summary())           # 절편항 없음

# 절편항 추가 후 회귀 모델링
df2_x = df_x[:]                   # deep copy
df2_x['intercept'] = 1            # X 데이터에 절편항 추가

m_reg4 = OLS(df_y, df2_x).fit()
print(m_reg4.summary())           # 절편항 추가
  

# formula 입력 모델
from statsmodels.formula.api import ols

df_x.columns                      # 'Temperature', 'Pressure', 'Time'
m_reg5 = ols('Strength ~ Temperature + Pressure + Time', data = df1).fit()
print(m_reg5.summary())



# 4. 회귀분석 결과
# 모형 유의, 설명력 높은 수준, 
# Temperature, Pressure > Strength 예상하지만
# Temperature 변수는 유의하지 않은 것을 확인

#    1) Temperature 제거 고려
#    2) 서로 독립적인 두 변수 Temperature, Pressure Interaction effect
df1['TempXPress'] = df_x['Temperature'] * df_x['Pressure']

m_reg6 = ols('Strength ~ Temperature + Pressure + Time + TempXPress', data = df1).fit()
print(m_reg6.summary()) 



# 5. 잔차 시각화(독립성, 등분산성, 정규성)
m_reg6.fittedvalues               # 적합값(예측값)
m_reg6.resid                      # 잔차항(실제값-예측값)

#    1) 독립성, 등분산성 검정
plt.scatter(m_reg6.fittedvalues, m_reg6.resid)   # 전차 산점도
plt.hist(m_reg6.resid)

# durbin_watson test
from statsmodels.stats.stattools import durbin_watson
durbin_watson(m_reg6.resid)                      # 1.61
# 설명변수 수(4), 훈련데이터 수(29)
# > Durbin-Watson Table를 이용하여 해당 값에 대한 해석이 필요함
#   [1.124, 1.743] 양쪽 구간에 가까울수록 자기상관이 강함
#                 (중간값이 가까울수록 자기 상관이 없음)


#    2) 잔차항에 대한 이상치 검정
#       H0 : 각 관측치가 이상치가 아님
#       H1 : 각 관측치가 이상치임
m_reg6.outlier_test()

# bonf(p) : bonferroni p-value
# 한 번에 여러 가설에 대한 가설 검정 수행 시 각 관측치별 가설검정 수행 결과로 재해석한 유의확률

# 이상치 잔차 확인
m_reg6.outlier_test()['bonf(p)'] < 0.05   # 이상치 없음


#    3) 정규성 검정
#       잔차 평균 0, 분산 σ²(일정)인 정규분포를 따름
plt.hist(m_reg6.resid)                    # 잔차 히스토그램

import seaborn as sb
sb.distplot(m_reg6.resid)                 # hist + kde

from scipy import stats
stats.probplot(m_reg6.resid, plot = plt)  # qq plot

# shapiro test
# H0 : 정규분포를 따름
# H1 : 정규분포를 따르지 않음
stats.shapiro(m_reg6.resid)
# pvalue = 0.86 > H0(정규분포를 따름) 채택



# 6. 다중공선성 진단
#    VIF = 1 / (1-R**2)
#    1) VIF 직접 계산
#    case 1) x1 ~ x2 + x3
for1 = 'Temperature ~ Pressure + Time'
m_reg11 = ols(for1, data = df_x).fit()
1 / (1-m_reg11.rsquared)                  # 1.12

#    case 2) x2 ~ x1 + x3
for2 = 'Pressure ~ Temperature + Time'
m_reg12 = ols(for2, data = df_x).fit()
1 / (1-m_reg12.rsquared)                  # 1.20

#    case 3) x3 ~ x1 + x2
for3 = 'Time ~ Temperature + Pressure'
m_reg13 = ols(for3, data = df_x).fit()
1 / (1-m_reg13.rsquared)                  # 1.09


#    2) VIF 코드 구현
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

df1 = pd.read_csv('data/Interactions_Continuous.csv')
y, X = dmatrices('Strength ~ Temperature + Pressure + Time', 
                 data = df1, return_type = 'dataframe')
vif(X.values, 0)                          # 절편에 대한 VIF : 1713.7285
vif(X.values, 1)                          # X1에 대한 VIF  : 1.1217
vif(X.values, 2)                          # X2에 대한 VIF  : 1.2058
vif(X.values, 3)                          # X3에 대한 VIF  : 1.0907



# 7. 다중공선성 문제 해결
#    1) 변수 제거 고려

#    2) PCA

#    3) 릿지, 라쏘, 엘라스틱넷
#       - Ridge
#         L2 규제 : 회귀 계수 제어(가중치의 제곱합을 최소화하는 방식으로 규제)
#         alpha : 규제정도를 조절하는 초매개변수(클수록 회귀계수 작아짐)
#       - Lasso
#         L1 규제 : 회귀 계수 제어(가중치의 절댓값의 합을 최소화하는 방식으로 규제)
#         alpha : 규제정도를 조절하는 초매개변수(클수록 회귀계수 작아짐, 0이 되기도-변수삭제 효과)
#       - ElasticNet : Ridge + Lasso


# 예제) boston 주택 가격에 대한 다중공선성 문제 해결
# 1. sklearn
from sklearn.linear_model import LinearRegression as reg


# 2. Ridge
from sklearn.linear_model import Ridge

m_rid1 = Ridge(alpha = 1).fit(boston_x, boston_y)
m_rid2 = Ridge(alpha = 0.1).fit(boston_x, boston_y)
m_rid3 = Ridge(alpha = 0.01).fit(boston_x, boston_y)
# > 추정된 회귀 계수가 0.001보다 작은 변수의 수


# 3. Lasso
from sklearn.linear_model import Lasso

m_ras1 = Lasso(alpha = 1).fit(boston_x, boston_y)
m_ras2 = Lasso(alpha = 0.1).fit(boston_x, boston_y)
m_ras3 = Lasso(alpha = 0.01).fit(boston_x, boston_y)
# > 추정된 회귀 계수가 0인 변수의 수





