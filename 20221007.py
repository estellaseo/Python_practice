# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 회귀분석
# =============================================================================
# [ 전처리 ]
# 1. 결측치 처리
#    1) 행 또는 컬럼 삭제 : 
#       - 한 컬럼에서 80% 이상의 데이터가 결측치인 경우
#       - 행의 경우도 일정 부분 이상의 컬럼값이 결측치인 경우

#    2) 결측치 대치 :
#       - 특정값으로 대치(최대, 최소, 평균 등으로 대치)
#       - 보간법(시계열에 따른 예측치로 대치, 머신러닝 기법을 사용한 대치)


# 2. 이상치 처리
#    1) 이상치 정의
#       - 과거 이상치 연구를 통해 이상치 기준 마련
#       - boxplot, 회귀모델의 경우 예측오차가 큰 값을 이상치로 판단
#    2) 처리 방법
#       - 대치 : 확실한 이상치에 대한 구간이 있을 때 최대(상한값), 최소(하한값)
#       - 삭제 : 완벽하게 구간 밖으로 과하게 벗어난 관측치의 경우 삭제를 고려(신중하게)


# 3. Label Encoding
#    - 문자값을 갖는 변수를 숫자로 대치
#    - ordered factor의 경우(ex. 상중하) 순서에 맞게 가중치를 반영할 수 있는 숫자로 변환


# 4. 변수 선택
#    - 도메인에 대한 이해를 기반으로 변수를 직접 제거 혹은 선택(전문가의 견해)
#    - 여러 변수 선택 기법을 활용한 최적의 변수 조합 결정
#      (전진선택법, 후진제거법, stepwise, 변수중요도(tree), 회귀계수(regression))


# 5. 변수 변환
#    - 로그 변환 : counting이 가능한 이산형 변수에 필수적으로 적용(ex. 방문 횟수, 성공 횟수, etc.)
#    - 제곱 변환
#    - 루트 변환


# 6. 파생 변수
#    - 도메인 지식이 없을 경우 까다로울 수 있음
#    - ex. 시간대비 방문횟수, 지출금액/방문횟수 등


# 7. Binning 변환
#    - 연속형 변수 구간화(범주형 변수로 변환) > 평활화


# [ 최적 모델 선택 ] 최적화 포함
# - Linear Regression
# - DT, RF, GB, LGB, EGB
# - KNN Regression
# - SVM
# - 신경망 기반




# [ 연습 문제 ] Sales 예측 - 회귀
# 1. 데이터 불러오기
df_train = pd.read_csv('data/bigmart_train.csv')
df_test = pd.read_csv('data/bigmart_test.csv')

Series(df_train['Item_Identifier'].unique())


# 2. 데이터 컬럼 살펴보기
df_train.columns
df_test.columns


# 3. train, test dataset 합치기
#    : 결측치, 이상치를 같은 기준으로 정의/처리하기 위해
df_test['Item_Outlet_Sales'] = 0
df_total = pd.concat([df_train, df_test])


# 4. 데이터 타입 확인
#    : 향후 데이터 처리 방향 계획(제거, 변환 고려 등)
df_total.dtypes

df_total['Item_Identifier']
# 기본적으로 ID 속성은 제거 고려하는 경우가 많지만 
# 이 경우 item 분류 코드로 활용 가능할 것으로 예상됨



# 5. 결측치
#    각 컬럼별 결측치 수/비율 확인
df_total.isna().sum()
len(df_total['Item_Weight'])              # 14204
 
df_train.isna().sum()
df_test.isna().sum()

df_total.isna().sum()/df_total.shape[0]
# Item_Weight, Outlet_Size 컬럼의 결측치 대치가 필요할 것으로 확인


# 문자형 변수 처리
df_total['Outlet_Size'].unique()          # ['Medium', nan, 'High', 'Small']
df_total['Item_Fat_Content'].unique()     # ['Low Fat', 'Regular', 'low fat', 'LF', 'reg']
df_total['Item_Type'].unique()            # ['Dairy', 'Soft Drinks', 'Meat' ...]
df_total['Outlet_Location_Type'].unique() # ['Tier 1', 'Tier 3', 'Tier 2']
df_total['Outlet_Type'].unique()          # ['Supermarket Type1', 'Supermarket Type2', 
                                          #  'Grocery Store', 'Supermarket Type3']


# 데이터 처리
# 1) 명목형, 순서형 등 그에 따른 숫자 변환
# 2) Item_Fat_Content 컬럼에 중복 데이터 값 통일시켜야함
#    ['Low Fat', 'Regular', 'low fat', 'LF', 'reg'] > ['LF', 'reg']



# 결측치 처리
# 1. Outlet_Size
#    case 1) Outlet_Location_Type으로 Outlet_Size 유추?
#    case 2) Outlet_Identifier으로 Outlet_Size 유추?
#    case 3) Outlet_Type으로 Outlet_Size 유추?
df_total['Outlet_Identifier'].unique()
df_total.loc[df_total['Outlet_Identifier'] == 'OUT049', 'Outlet_Size'].unique()

df_total.loc[df_total['Outlet_Size'].isnull(), 'Outlet_Identifier'].unique()

df_total.loc[df_total['Outlet_Identifier'] == 'OUT017', 'Outlet_Size'].unique()


# 2. Item_Weight
#    
df_total['Item_Weight'] = df_total['Item_Weight'].fillna(df_total['Item_Weight'].mean())

    
# 6. Label Encoding
from sklearn.preprocessing import LabelEncoder
m_le = LabelEncoder()
m_le.fit(df_total['Item_Fat_Content'])        # unique value 찾기
m_le.transform(df_total['Item_Fat_Content'])  # unique calue 별로 서로 다른 숫자 변환

#    문자형 변수 목록 추출
char_cols = df_total.dtypes[df_total.dtypes == 'object'].index

for i in char_cols :
    df_total[i] = m_le.fit_transform(df_total[i])

# 데이터타입 최종 확인
df_total.dtypes


# 7. 모델링
# 1) sklearn linear reg
df_raw = df_total.loc[df_total['Item_Outlet_Sales'] != 0, :]
df_raw_x = df_raw.drop('Item_Outlet_Sales', axis = 1)
df_raw_y = df_raw['Item_Outlet_Sales']

from sklearn.linear_model import LinearRegression as reg
m_reg = reg()
m_reg.fit(df_raw_x, df_raw_y)
m_reg.score(df_raw_x, df_raw_y)        # 0.5019

# > 이상치, 결측치, 변수 선택에 민감한 모델
# > 연속형 변수를 이용한 예측에 좋은 성능을 가진 모


# 2) OLS
from statsmodels.api import OLS


# 절편항 추가 후 회귀 모델링
df2_x = df_raw_x[:] 
df2_x['intercept'] = 1 

m_reg4 = OLS(df_raw_y, df2_x).fit()
print(m_reg4.summary())                # 절편항 추가

# 3) rf_r
from sklearn.ensemble import GradientBoostingRegressor as gb_r
m_reg3 = gb_r()
m_reg3.fit(df_raw_x, df_raw_y)
m_reg3.score(df_raw_x, df_raw_y)       # 62.95






