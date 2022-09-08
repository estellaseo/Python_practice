# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# SVM(Support Vector Machine)
# =============================================================================
# 지도 학습 > 분류 모델
# 주어진 데이터를 분류하기 위한 규칙을 찾는 기법(회귀 지원 가능)
# 분류 기준(결정경계, 초평면)을 찾는 것이 목적
# 예측력이 강하고 과적합되기 쉽지 않은 모델
# 모델을 해석하기 어려움
# 학습 변수에 영향을 많이 받음(사전 변수 선택 중요)
# 이상치에 민감
# 변수 스케일링 필요

# 서포트 벡터 (Support Vector)
# 결정 경계에 가까운 데이터들의 집합(가중치 높임)


# 예제) SVM hyperplane(초평면) 유도과정
# 1. Data loading
from sklearn.datasets import make_blobs
X, y = make_blobs(centers = 4, random_state = 8)

X.shape                          # (100, 2)
y.shape                          # (100,)

y = y % 2                        # y의 클래스가 2개인 경우를 가정하기 위해 y값 변환


# 2. 데이터 분포 확인
pip install mglearn
import mglearn                   # 시각화 패키지
mglearn.discrete_scatter(X[:, 0],# x축 좌표 
                         X[:, 1],# y축 좌표 
                         y)      # y값에 따른 색상 구분


# 3. 선형 분류기를 통한 분류 시도
#    발생 가능한 모든 결정 경계 중 마진을 최대로 하는 결정 경계 선택
from sklearn.svm import LinearSVC
m_svm = LinearSVC()
m_svm.fit(X, y)                  # Warning : n_features(설명변수 수) < n_samples

m_svm.intercept_                 # 유도된 결정경계의 절편
m_svm.coef_                      # 유도된 결정경계의 기울기

mglearn.plots.plot_2d_separator(m_svm,  # 결정경계를 갖는 모델명
                                X)      # 설명변수 dataset


# 4. 3차원 데이터로 변경 후 시각화
#    기존에 가지고 있는 설명변수를 변경하여 새로운 파생변수 유도

X[:, 1 ]**2                      # 두번째 설명변수의 제곱
X_new = pd.concat([DataFrame(X), DataFrame(X[:, 1]**2)], axis = 1).values

# [ 참고 ]
np.hstack()                      # 가로 방향 데이터 결합
np.vstack()                      # 세로 방향 데이터 결합

np.hstack([X, X[:, 1]**2])       # 에러 발생(2차원과 1차원 결합 불가)
np.hstack([X, X[:, 1:2]**2])     # 1차원 데이터를 2차원으로 변경 후 결합


# 시각화
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d  # 3차원 공간선상 시각화

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X_new[y == 0, 0],     # x축 좌표: y값이 0인 데이터의 첫번째 컬럼
           X_new[y == 0, 1],     # y축 좌표
           X_new[y == 0, 2],     # z축 좌표
           c = 'b',              # 점 색상
           cmap = mglearn.cm2,   # 팔레트
           s = 60,               # 점 크기
           edgecolors = 'k')     # 점 테두리 색

ax.scatter(X_new[y == 1, 0],
           X_new[y == 1, 1],
           X_new[y == 1, 2],
           c = 'r'
           cmap = mglearn.cm2,
           s = 60, 
           edgecolors = 'k') 


# 5. 초평면(분류평면) 유도 및 시각화
m_svm2 = LinearSVC()
m_svm2.fit(X_new, y)

intercept = m_svm2.intercept_ 
coef = m_svm2.coef_.ravel()

# 시각화
np.linspace(-3, 3, 1000)         # -3부터 3까지 1000개로 균등분할(x축 좌표를 위해)

xx = np.linspace(X_new[:, 0].min()-2, X_new[:, 0].max()+2, 1000)
yy = np.linspace(X_new[:, 1].min()-2, X_new[:, 1].max()+2, 1000)

XX, YY = np.meshgrid(xx, yy)     # 2차원 공간 안의 좌표값으로 변환

ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]

ax.plot_surface(XX, YY, ZZ, alpha = 0.3)



# [ 연습 문제 ] cancer data 암 종양 분류(SVM)
from sklearn.svm import SVC

# 스케일링 전
m_svm3 = SVC()
m_svm3.fit(train_x, train_y)
m_svm3.score(train_x, train_y)   # 0.9084507042253521
m_svm3.score(test_x, test_y)     # 0.958041958041958

# 스케일링 후
m_svm3 = SVC()
m_svm3.fit(train_x_sc, train_y)
m_svm3.score(train_x_sc, train_y) # 0.9859154929577465
m_svm3.score(test_x_sc, test_y)   # 0.972027972027972




# =============================================================================
# 변수 스케일링
# =============================================================================
# 각 설명변수가 갖는 범위가 다를 때 동일 선상에서 비교하기 위함
# 거리 기반 모델, 회귀 기반 모델, NN 모델 등에서 필요
# 회귀 계수의 비교(변수가 갖는 스케일에 따라 계수의 크기가 달라질 수 있음)

# 1. standard scale
#    평균과 표준편차를 이용한 변수 조절
# 2. minmax scale
#    최소, 최대를 이용한 변수 조절
# 3. robust scale
#    사분위수를 이용한 변수 조절


# step 1) 모듈 호출
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.preprocessing import RobustScaler as robust


# step 2) 스케일링
m_sc = standard()
m_sc.fit(train_x)                # 각 컬럼별 평균, 표준편차 계산

train_x_sc = m_sc.transform(train_x)          # 각 데이터 변환
test_x_sc = m_sc.transform(test_x)            # 주의: 같은 기준으로 변환해야 함



# [ 스케일링 결과 시각화 ]
# 서로 다른 기준으로 scaling한 결과(잘못된 scaling)
m_sc2 = standard()
m_sc2.fit(test_x)
test_x_sc2 = m_sc2.transform(test_x)

fig, ax = plt.subplots(1, 3)     # fig : figure 이름, ax = subplot 전체명
plt.rc('font', family = 'Malgun Gothic')      # 글꼴 변경


# 1) 원본 산점도
ax[0].scatter(train_x[:, 0], train_x[:, 1], c = mglearn.cm2(0), label = 'train') 
ax[0].scatter(test_x[:, 0], test_x[:, 1], c = mglearn.cm2(1), label = 'test') 
ax[0].legend()
ax[0].set_title('원본 산점도')


# 2) 올바른 스케일링 결과 산점도
ax[1].scatter(train_x_sc[:, 0], train_x_sc[:, 1], 
              c = mglearn.cm2(0), label = 'train_scaling') 
ax[1].scatter(test_x_sc[:, 0], test_x_sc[:, 1], 
              c = mglearn.cm2(1), label = 'test_scaling') 
ax[1].legend()
ax[1].set_title('올바른 스케일링 산점도')


# 3) 잘못된 스케일링 결과 산점도
ax[2].scatter(train_x_sc[:, 0], train_x_sc[:, 1], 
              c = mglearn.cm2(0), label = 'train_scaling2') 
ax[2].scatter(test_x_sc2[:, 0], test_x_sc2[:, 1], 
              c = mglearn.cm2(1), label = 'test_scaling2')
ax[2].legend()
ax[2].set_title('잘못된 스케일링 산점도') 
# > 데이터가 기존의 특성을 그대로 유지한다고 보기 어려움
