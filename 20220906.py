# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 분석의 형태
# =============================================================================
# 1. 지도학습(Supervised Learning)
#    - Y가 존재하는 분석 형태(예측 모델링 기법)
#    - 회귀분석(LR(Linear Regression), ridge, lasso, Elastic-Net)
#    - 분류분석(로지스틱회귀, DT, RF, GB, XGB, SVM, knn)


# 2. 비지도학습(Unsupervised Learning)
#    - Y가 존재하지 않는 분석 형태
#    - 데이터 세분화(클러스터링), 데이터포인트끼리의 연관 규칙 발견
#    - 군집분석(계층적, 비계층적(k-means)), 연관분석, PCA




# =============================================================================
# 의사결정나무
# =============================================================================
# - 지도학습 > 분류모델 > 트리기반모델
# - 이상치와 결측치에 비교적 덜 민감
# - 변수 스케일링 필요하지 않음
# - 설명변수의 데이터가 연속형, 이산형, 범주형일 때도 효과적임
# - 변수 중요도 계산(불순도 기반) > 변수 선택 시에 사용
# - train dataset에 영향을 많이 받음 > 과대, 과소 적합되기 쉬움




# =============================================================================
# 의사결정나무 실습
# =============================================================================
# 1. Decision Tree Model
# step 1) data loading
from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()                              # sklearn에서 제공하는 sample dataset
#                                          : dictionary 형식으로 제공

# 변수(설명) = feature = 특성 = 속성 = 차원
iris['data']                             # 설명변수 데이터
iris['target']                           # 종속변수 데이터
iris['target_names']                     # target 이름 데이터

iris_x = iris['data']
iris_y = iris['target']


# step 2) import modules
dir(sklearn.tree)
from sklearn.tree import DecisionTreeClassifier as dt_c


# step 3) data split
from sklearn.model_selection import train_test_split

train_test_split(*iterables,             # 분리할 dataset(x data, y data)
                 test_size = 0.25,       # test dataset size
                 train_size = 0.75,      # train dataset size
                 random_state)           # seed 값

train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, 
                                                    random_state = 0)


# step 4) modeling
m_dt = dt_c()                            # 빈 모델 (매개변수 정의)
m_dt.fit(train_x, train_y)               # 학습

m_dt.feature_importances_                # 변수 중요도


# step 5) score
(m_dt.predict(test_x) == test_y).sum()/len(test_y)
m_dt.score(test_x, test_y)               # 0.9736842105263158


# step 6) tuning
# - max_depth : 최대 길이
# - min_samples_split : 최소 가지치기 기준(오분류 개수 설정)
# - max_features : 후보 설명변수의 수(임의성 정도)

score_tr = []; score_te = []
for i in range(2, 11) :
    m_dt = dt_c(min_samples_split = i)
    m_dt.fit(train_x, train_y)
    score_tr.append(m_dt.score(train_x, train_y))
    score_te.append(m_dt.score(test_x, test_y))
    
score_tr
score_te


# step 7) predict
new_data = [[5.7, 3.1, 2.8, 0.8]]
iris['target_names'][m_dt.predict(new_data)][0]




# step 9) visualization
# 1) graphviz 설치(window)
# download => https://graphviz.gitlab.io/_pages/Download/Download_windows.html    
# download 후 설치(64bit exe file)

# 2) graphviz 설치(Python)
pip install graphviz

# 3) Python path 설정
import os
os.environ['PATH'] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# 4) graphviz를 이용한 시각화
import graphviz
from sklearn.tree import export_graphviz

export_graphviz(m_dt,                    # 시각화할 모델명
                out_file = ,             # 저장할 파일
                class_names = ,          # 종속변수 이름
                feature_names = ,        # 설명변수 이름
                impurity = True,         # 불순도 출력 여부
                filled = False)          # 노드 이름 출력 여부

export_graphviz(m_dt,                  
                out_file = 'tree.dot',   
                class_names = iris['target_names'],           
                feature_names = iris['feature_names'],         
                impurity = False,      
                filled = True)        

with open('tree.dot', encoding = 'UTF8') as f :
    dot_graph = f.read()

g1 = graphviz.Source(dot_graph)
g1.render('dt_1', cleanup = True) 


export_graphviz()




