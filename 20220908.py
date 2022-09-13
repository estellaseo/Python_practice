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

X[:, 1 ]**2                      # 두번째 설명변수의 제곱 > 커널 트릭
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




# 예제) iris 꽃 분류(svm)
# 1. 데이터 로딩
from sklearn.datasets import load_iris
iris = load_iris()

iris_x = iris['data']
iris_y = iris['target']


# 2. 변수 스케일링
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax

m_sc = minmax()
m_sc.fit(iris_x)
iris_x_sc = m_sc.transform(iris_x)

iris_x_sc.min(axis = 0)          # 스케일링 후 최솟값 (0)
iris_x_sc.max(axis = 0)          # 스케일링 후 최댓값 (1)


# 3. 데이터 분리
from sklearn.model_selection import train_test_split
train_sc_x, test_sc_x, train_y, test_y = train_test_split(iris_x_sc, iris_y,
                                                                 random_state = 0)

# 4. 모델링
from sklearn.svm import SVC
m_svm = SVC()
m_svm.fit(train_sc_x, train_y)

m_svm.score(train_sc_x, train_y) # 0.9642857142857143
m_svm.score(test_sc_x, test_y)   # 0.9736842105263158


# 5. 매개변수(하이퍼 파라미터) 튜닝
# 1) C
#    - 슬랙변수(오분류 데이터와 결정경계의 마진을 표현하는 변수)의 조절값
#    - 오분류 데이터포인트에 주는 가중치(클수록 복잡한 경계)
#    - 비선형성 강화 요인

# 2) gamma
#    - 고차원으로의 매핑 정도를 나타내는 변수
#    - 고차원 정도 강화 요인(클수록 복잡한 경계)
#    - 결정경계 인근에 있는 데이터포인트 가중치 부여
#    - gamma 값이 클수록 보다 결정경계와 가까운 데이터포인트에 집중 

v_C = [0.001, 0.01, 0.1, 1, 10, 100, 10000]
v_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 10000]

# case 1) best score, best parameter 확인
best_score = 0
for i in v_C:
    for j in v_gamma:
        m_svm = SVC(C = i, gamma = j)
        m_svm.fit(train_sc_x, train_y)
        vscore_te = m_svm.score(test_sc_x, test_y)
        
        if vscore_te > best_score : 
            best_params = {'C': i, 'gamma':j}
            best_score = vscore_te
            
# 결과
best_score                       # 0.9736842105263158
best_params                      # {'C': 0.1, 'gamma': 10}


# case 2) 모든 파라미터 변화에 따른 예측점수 확인
vscore_tr = []; vscore_te = []; v_i = []; v_j = []
for i in v_C:
    for j in v_gamma:
        m_svm = SVC(C = i, gamma = j)
        m_svm.fit(train_sc_x, train_y)
        v_i.append(i)
        v_j.append(j)
        vscore_tr.append(m_svm.score(train_sc_x, train_y))
        vscore_te.append(m_svm.score(test_sc_x, test_y))

# C, gamma 변화에 따른 점수 변화 plot 시각화
import matplotlib.pyplot as plt
plt.plot(vscore_tr, c = 'r', label = 'vscore_tr')
plt.plot(vscore_te, c = 'b', label = 'vscore_te')
plt.legend()

# C, gamma 변화에 따른 점수 변화 데이터프레임화
DataFrame({'C':v_i, 'gamma':v_j, 'tr_score':vscore_tr, 'te_score':vscore_te})

# C, gamma 변화에 따른 점수 변화 heatmap 시각화
from mglearn.tools import heatmap
heatmap(values = ,                            # 표현할 2차원 array
        xlabel = ,                            # x축 이름
        ylabel = ,                            # y축 이름
        xticklabels = ,                       # x축 눈금
        yticklabels = ,                       # y축 눈금
        cmap)                                 # 팔레트

plt.summer()                                  # 현 세션의 기본 컬러맵 변경
ascore_te = np.array(vscore_te).reshape(7, 7) # 행 우선순위로 배열 완성
heatmap(ascore_te, 'gamma', 'C', v_C, v_gamma, 'Pastel2_r')

# [ 참고 ] 팔레트 확인 방법
plt.colormaps()

# 6. 최종모델 만들기
m_svm = SVC(C = 0.1, gamma = 10)
m_svm.fit(train_sc_x, train_y)

m_svm.predict()


# 예제) cancer data를 사용한 종양 양성/악성 분류 SVM
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer_x = cancer['data']
cancer_y = cancer['target']
cancer_x_df = DataFrame(cancer_x, columns = cancer['feature_names'])


# 1. DT/RF를 사용한 변수 중요도 확인(top 2)
m_dt = dt_c()
m_dt.fit(cancer_x, cancer_y)

s1 = Series(m_dt.feature_importances_, index = cancer['feature_names'])
s1.sort_values(ascending = False).index[:2]


# 2. 위 두 변수를 이용한 2차원 산점도 시각화
import mglearn
cancer_x_df2 = cancer_x_df[['worst radius', 'worst concave points']]
mglearn.discrete_scatter(cancer_x_df2.iloc[:,0], 
                         cancer_x_df2.iloc[:,1], 
                         cancer_y)
plt.xlabel('worst radius')
plt.ylabel('worst concave points')


# 3. 위 두 변수를 이용한 SVM 모델(선형) 적용
from sklearn.svm import LinearSVC
m_svm = LinearSVC()
m_svm.fit(cancer_x_df2, cancer_y)
mglearn.plots.plot_2d_separator(m_svm, cancer_x_df2) 


# 4. 적용 결과 시각화
m_sc = minmax()
cancer_x_df2_sc = m_sc.fit_transform(cancer_x_df2)

m_svm = LinearSVC()
m_svm.fit(cancer_x_df2_sc, cancer_y)
mglearn.discrete_scatter(cancer_x_df2_sc[:,0], cancer_x_df2_sc[:,1], cancer_y)

mglearn.plots.plot_2d_separator(m_svm, cancer_x_df2_sc)    
plt.ylim(cancer_x_df2_sc[:,1].min() - 0.1, cancer_x_df2_sc.max() + 0.1)




# [ SVM 용어 정리 ]
# - 결정경계 : 데이터를 분류하는 분류 선 or 평면
# - 초평면 : 다차원일때의 분류 경계를 나타내는 평면
# - Support Vector : 결정경계에 인접한 데이터포인트
# - 마진 : 결정경계로부터의 거리(마진을 최대화하는 분류경계 생성)
# - 슬랙 변수 : 오분류된 데이터의 오차를 표현하는 방식
#              선형 분리가 불가능할 경우 개별적인 오차를 허용하여 결정경계 생성
#              슬랙변수 = 0 > 선형으로 완벽하게 분리됨
# - 커널(Kernel) : 주어진 데이터를 고차원 공간으로 매핑시켜주는 함수
#                 ex) 제곱변환, 시그모이드변환




# [ 장점 ]
# - 예측력이 강함
# - 저차원 데이터로도 좋은 예측 결과 리턴

# [ 단점]
# - 연산 비용이 높음
# - 큰 데이터에 적합하지 않음(100,000 데이터 이상의 경우 분석이 매우 느림)
# - 분석과 해석이 어려움
# - 변수의 스케일에 영향을 많이 받음
# - 결측치, 이상치에 매우 민감
# - 연속형 설명변수로만 구성됐을 때 가장 효과적




