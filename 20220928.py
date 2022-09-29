# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 데이터 불균형 처리
# =============================================================================
# 분류 과제일 때, Y의 각 클래스별 빈도수의 차이가 심할 때 예측 정확도의 신뢰도가 떨어짐
# 불균형 데이터 처리를 수행하면 정밀도 향상
# 예) 100명 중 합격 97명, 불합격 3명
#     > 100명 모두 합격이라 하더라도 97%의 예측률이라고 판단할 수 있음


# 1. 언더샘플링
#    - 다수 클래스를 소수 클래스에 맞게 사이즈를 줄이는 방식
#    - 데이터의 소실 발생(유의미한 데이터가 생략될 수 있음)

#    1) 랜덤 언더 샘플링
#       : 무작위로 다수 클래스 일부 추출
#    2) ENN(Edited Nearest Neighbors)
#       : 소수 클래스 주변에 위치한 다수 클래스 데이터 제거
#    3) CNN(Condensed Nearest Neighbors)
#       : 다수 클래스에 밀집된 데이터를 제거하여 대표만 남기는 방식
#    4) 토멕링크방법
#       : 서로 다른 클래스에 속하면서 위치가 가까운 한 쌍의 데이터 중 
#         다수 클래스에 속하는 데이터를 제거하는 방식
#    5) OSS(One Sided Selection)
#       : 토멕링크 방법과 CNN 기법의 장점을 혼합
#         다수 클래스를 토멕링크 방법으로 제거 후 CNN을 사용하여 밀집 데이터 제거
#         앙상블 기법에 속함


# 예시) 비대칭 데이터에 대한 언더샘플링
# 1. 사전 준비(패키지 설치 및 로딩)
# pip install imblearn
# pip show scikit-learn
# pip show scikit-learn==1.1.0
import imblearn.under_sampling
dir(imblearn.under_sampling)

from imblearn.under_sampling import EditedNearestNeighbours as enn
from imblearn.under_sampling import RepeatedEditedNearestNeighbours as renn
from imblearn.under_sampling import CondensedNearestNeighbour as cnn
from imblearn.under_sampling import TomekLinks as tomek
from imblearn.under_sampling import OneSidedSelection as oss
from imblearn.under_sampling import RandomUnderSampler as random_s


import matplotlib.pyplot as plt


# 2. 데이터 불러오기
df1 = pd.read_csv('data/ThoraricSurgery.csv')
df1.columns

X = df1.drop('Risk1Yr', axis = 1)
y = df1['Risk1Yr']

# y 클래스별 비율 확인
y.value_counts()                              # 400, 70(뷸균등 확인)
y.value_counts().plot(kind = 'bar')           # 시각화


# 3. 언더샘플링
# 1) Random Sampling
m_random_s = random_s()
X_resampled, y_resampled = m_random_s.fit_resample(X, y)

y_resampled.value_counts()                    # 70, 70(균등 확인)
y_resampled.value_counts().plot(kind = 'bar') # 완벽하게 1:1로 변화

# 2) ENN
m_enn = enn()
X_resampled1, y_resampled1 = m_enn.fit_resample(X, y)

y_resampled1.value_counts()                    # 241, 70(뷸균등 확인)
y_resampled1.value_counts().plot(kind = 'bar') # 완벽하게 1:1로 변화되지는 않았음

# RENN
m_renn = renn()
X_resampled2, y_resampled2 = m_renn.fit_resample(X, y)

y_resampled2.value_counts()                    # 153, 70(뷸균등 확인)
y_resampled2.value_counts().plot(kind = 'bar')


# 3) CNN
m_cnn = cnn(random_state = 0)
X_resampled3, y_resampled3 = m_cnn.fit_resample(X, y)

y_resampled3.value_counts()                    # 128, 70(뷸균등 확인)
y_resampled3.value_counts().plot(kind = 'bar')


# 4) Tomeklink
m_tomek = tomek()
X_resampled4, y_resampled4 = m_tomek.fit_resample(X, y)

y_resampled4.value_counts()                    # 368, 70(뷸균등 확인)


# 5) OSS
m_oss = oss(random_state=0)
X_resampled5, y_resampled5 = m_oss.fit_resample(X, y)

y_resampled5.value_counts()                    # 366, 70(뷸균등 확인)


# 4. 언더샘플링 비교
# - PCA로 차원 축소(2개 변수 유도)
from sklearn.preprocessing import StandardScaler as standard
from sklearn.decomposition import PCA

m_sc = standard()
m_sc.fit(X)
X_sc = m_sc.transform(X)

m_pca = PCA(2)
m_pca.fit(X_sc)

X_resampled_sc = m_sc.transform(X_resampled)
X_resampled_sc_pca = m_pca.transform(X_resampled_sc)

X_resampled_sc1 = m_sc.transform(X_resampled1)
X_resampled_sc_pca1 = m_pca.transform(X_resampled_sc1)

X_resampled_sc2 = m_sc.transform(X_resampled2)
X_resampled_sc_pca2 = m_pca.transform(X_resampled_sc2)

X_resampled_sc3 = m_sc.transform(X_resampled3)
X_resampled_sc_pca3 = m_pca.transform(X_resampled_sc3)

X_resampled_sc4 = m_sc.transform(X_resampled4)
X_resampled_sc_pca4 = m_pca.transform(X_resampled_sc4)

X_resampled_sc5 = m_sc.transform(X_resampled5)
X_resampled_sc_pca5 = m_pca.transform(X_resampled_sc5)


# - 2차원 공간 산점도
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 6)

ax[0].scatter(X_resampled_sc_pca[:, 0], X_resampled_sc_pca[:, 1], c = y_resampled)
ax[0].set_title('Random Sampling')

ax[1].scatter(X_resampled_sc_pca1[:, 0], X_resampled_sc_pca1[:, 1], c = y_resampled1)
ax[1].set_title('ENN')

ax[2].scatter(X_resampled_sc_pca2[:, 0], X_resampled_sc_pca2[:, 1], c = y_resampled2)
ax[2].set_title('RENN')

ax[3].scatter(X_resampled_sc_pca3[:, 0], X_resampled_sc_pca3[:, 1], c = y_resampled3)
ax[3].set_title('CNN')

ax[4].scatter(X_resampled_sc_pca4[:, 0], X_resampled_sc_pca4[:, 1], c = y_resampled4)
ax[4].set_title('Tomeklink')

ax[5].scatter(X_resampled_sc_pca5[:, 0], X_resampled_sc_pca5[:, 1], c = y_resampled5)
ax[5].set_title('OSS')



# [ 참고 ] 토멕링크 원리
from sklearn.neighbors import NearestNeighbors

# 1. 소수클래스 기준 각 관측치마다 가장 가까운(거리) 이웃 1개 확인
m_nn = NearestNeighbors(n_neighbors=2)
m_nn.fit(X)
# 나를 제외한 가장 가까운 이웃 index
nns = m_nn.kneighbors(X, return_distance = False)[:, 1] 

# 2. 해당 클래스가 서로 다른 클래스일 경우 다수 클래스에 속하는 데이터 제거
istomek = m_tomek.is_tomek(y, nns, [1]) # 실제 y값, 각 y값의 NN index, 소수클래스명
y.index[istomek]                        # 토멕링크 index
nns[istomek]                            # 각 토멕링크의 NN

DataFrame({'rownum' : y.index[istomek], 'NN' : nns[istomek]})



# 2. 오버샘플링
#    - 소수 클래스를 다수 클래스에 맞게 사이즈를 늘리는 방식
#    - 과적합이 위험
#    - 랜덤 오버 샘플링, SMOTE, Goardline-Am

#    1) 랜덤 오버 샘플링
#       : 무작위로 소수 클래스를 복제하여 데이터의 비율을 맞추는 방법(중복)
#    2) SMOTE(Synthetic Minority Over-sample TEchnique)
#       : 소수 클래스의 중심 데이터와 주변 데이터 사이의 선형추세를 반영하여 데이터 추가
#    3) Borderline-SMOTE
#       : 다수 클래스와 소수 클래스의 경계선에서 SMOTE 적용(선형추세 반영)
#    4) ADASYN(ADAptive SYNthetic)
#       : 모든 소수 클래스에서 다수 클래스의 관측비율을 계산하여 SMOTE 적용


# 예제) 오버샘플링
# 1. 사전 준비(패키지 로딩)
import imblearn.over_sampling
dir(imblearn.over_sampling)

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN


# 2. 데이터 로딩
import pandas as pd
df1 = pd.read_csv('data/ThoraricSurgery.csv')
df1.columns

X = df1.drop('Risk1Yr', axis = 1)
y = df1['Risk1Yr']

# y 클래스별 비율 확인
y.value_counts()                              # 400, 70(뷸균등 확인)
y.value_counts().plot(kind = 'bar')           # 시각화

# 3. 오버 샘플링
# 1) Random Sampling
m_random_o = RandomOverSampler()
x_resampled, y_resampled = m_random_o.fit_resample(X, y)

y_resampled.value_counts()                    # 400, 400(균등 확인)
y_resampled.value_counts().plot(kind = 'bar') # 완벽하게 1:1로 변화

# 2) SMOTE(이진 클래스일 때 보다 효과적임)
m_smote = SMOTE()
x_resampled2, y_resampled2 = m_smote.fit_resample(X, y)

y_resampled2.value_counts()                    # 400, 400(균등 확인)
y_resampled2.value_counts().plot(kind = 'bar')

# 3) borderline-SMOTE
m_bsmote = BorderlineSMOTE()
x_resampled3, y_resampled3 = m_bsmote.fit_resample(X, y)

y_resampled3.value_counts()                    # 400, 400(균등 확인)
y_resampled3.value_counts().plot(kind = 'bar')

# 4) ADASYN(다중 클래스일 때 보다 효과적임)
m_adasyn = ADASYN()
x_resampled4, y_resampled4 = m_adasyn.fit_resample(X, y)

y_resampled4.value_counts()                    # 400, 387(불균등 확인)
y_resampled4.value_counts().plot(kind = 'bar')


# 4. 오버샘플링 결과 비교(시각화)
# - PCA로 차원 축소(2개 변수 유도)
from sklearn.preprocessing import StandardScaler as standard
from sklearn.decomposition import PCA

m_sc = standard()
m_sc.fit(X)
X_sc = m_sc.transform(X)

m_pca = PCA(2)
m_pca.fit(X_sc)

x_resampled_sc = m_sc.transform(x_resampled)
x_resampled_sc_pca = m_pca.transform(x_resampled_sc)

x_resampled_sc2 = m_sc.transform(x_resampled2)
x_resampled_sc_pca2 = m_pca.transform(x_resampled_sc2)

x_resampled_sc3 = m_sc.transform(x_resampled3)
x_resampled_sc_pca3 = m_pca.transform(x_resampled_sc3)

x_resampled_sc4 = m_sc.transform(x_resampled4)
x_resampled_sc_pca4 = m_pca.transform(x_resampled_sc4)


# - 2차원 공간 산점도
import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')
plt.rc('axes', unicode_minus = False)
fig, ax = plt.subplots(1, 4)

ax[0].scatter(x_resampled_sc_pca[:, 0], x_resampled_sc_pca[:, 1], c = y_resampled)
ax[0].set_title('Random Sampling')

ax[1].scatter(x_resampled_sc_pca2[:, 0], x_resampled_sc_pca2[:, 1], c = y_resampled2)
ax[1].set_title('SMOTE')

ax[2].scatter(x_resampled_sc_pca3[:, 0], x_resampled_sc_pca3[:, 1], c = y_resampled3)
ax[2].set_title('Borderline SMOTE')

ax[3].scatter(x_resampled_sc_pca4[:, 0], x_resampled_sc_pca4[:, 1], c = y_resampled4)
ax[3].set_title('ADASYN')



# 3. 임곗값 이동
#    - 분류모델 생성 시 분류기준에 데이터 클래스의 비율을 반영하여 임계값을 조절

# ex) 카드사 사기거래 판별(class_0:class_1 = 9:1)
# class == 0 : 합법적 거래
# class == 1 : 사기로 판명된 거래













