# -*- coding: utf-8 -*-
from numpy import nan as NA
from pandas import Series, DataFrame

import numpy as np
import pandas as pd

# =============================================================================
# 이미지 인식
# =============================================================================
# 1. dataset 설명
# sklearn 유명인사 62명 얼굴 데이터(흑백)의 RGB 값
# 3,023개 제공 (87X65)

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person = 50)

people_x = people['data']
people_y = people['target']

people_x.shape                        # (3023, 5655) 3023개 이미지, 평탄화(87X65)
people_y                              # 각 인물의 이름 > 숫자 변환(0 ~ 61)
people['target_names']                # 각 인물의 실제 이름값

people_x[0, :]                        # RGB 변환된 값(0~255)
people_img = people['images']         # 평탄화 시키기 이전 데이터

plt.imshow(people_img[0, :, :])       # 첫번째 이미지 출력


# 2. 데이터 가공
#    1) 스케일링
people_x = people_x / 255             # min-max scaling 효과

#    2) NN 학습을 위한 Y 변환
people_y_dummy = pd.get_dummies(people_y).values


# 3. 데이터 분리
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y, train_y_dm, test_y_dm = train_test_split(people_x, 
                                                                           people_y, 
                                                                           people_y_dummy, 
                                                                           random_state=0)


# 4. 모델링
#    1)KNN으로 분류
from sklearn.neighbors import KNeighborsClassifier as knn_c
m_knn = knn_c()
m_knn.fit(train_x, train_y)
m_knn.score(train_x, train_y)         # 0.6128
m_knn.score(test_x, test_y)           # 0.4

from sklearn.metrics import confusion_matrix, classification_report

yhat = m_knn.predict(test_x)
print(classification_report(test_y, yhat, target_names=people['target_names']))


# cofusion matrix 시각화
mat = confusion_matrix(test_y, yhat)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(mat, cmap='summer')
ticks = np.arange(0,len(people['target_names']))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(people['target_names'], rotation=45, ha='right')
ax.set_yticklabels(people['target_names'], rotation=45, ha='right')
ax.set_ylabel('true label')
ax.set_xlabel('predicted label')
ax.xaxis.set_ticks_position('bottom')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, mat[i, j], ha='center', va='center')


#    2) PCA + knn
from sklearn.decomposition import PCA
m_pca = PCA(256)
train_x_pca = m_pca.fit_transform(train_x)
test_x_pca = m_pca.transform(test_x)

train_x_pca.shape                     # (1170, 256) 256개의 설명변수로 변경


# knn
m_knn = knn_c()
m_knn.fit(train_x_pca, train_y)
m_knn.score(train_x_pca, train_y)     # 0.6170
m_knn.score(test_x_pca, test_y)       # 0.4076



# 4. 예측
m_knn.predict(test_x) == test_y       # 맞춘게 거의 없음...(두번째 데이터 예측 실패)
testset = test_x[1:2, :]              # 두번째 데이터를 차원을 유지한 형태로 testset으로 저장
v_pre = m_knn.predict(testset)[0]     # 5번이라 예측
test_y[1]                             # 29번이 실제값

p1 = people['target_names'][v_pre]    # 'Ariel Sharon'
p2 = people['target_names'][test_y[1]]# 'Jeremy Greenstock' 



# 5. 예측값과 실제값 시각화 비교
img_pre = people['images'][people_y == 5, :, :][0,:,:] # 5의 target 값을 먼저 추출한 뒤 그 중 첫번째 이미지 선택
img_tru = people['images'][people_y == 29, :, :][0,:,:]# 29의 target 값을 먼저 추출한 뒤 그 중 첫번째 이미지 선택

fig, ax = plt.subplots(1,2)
ax[0].imshow(img_pre)
ax[1].imshow(img_tru)

ax[0].set_title(f'predict : {p1}')
ax[1].set_title(f'true : {p2}')




# 다음 인공변수 수의 변화에 따른 예측률 변화 확인
n_com = np.arange(100, 1100, 100)
tr_score = []; te_score = []

for i in n_com :
    m_pca = PCA(i)
    train_x_pca = m_pca.fit_transform(train_x)
    test_x_pca = m_pca.transform(test_x)

    m_knn = knn_c()
    m_knn.fit(train_x_pca, train_y)
    tr_score.append(m_knn.score(train_x_pca, train_y))
    te_score.append(m_knn.score(test_x_pca, test_y))
    
plt.plot(n_com, tr_score, c = 'r', label = 'train')
plt.plot(n_com, te_score, c = 'b', label = 'train')
plt.ylim([0.3, 0.9])
plt.xlabel('n_components')
plt.ylabel('accuracy')
plt.legend()



# [ 참고 ] 이미지 변환(RGB값 추출)
import imageio
img1 = imageio.imread('joshua.jfif')
img1.shape

import matplotlib.pyplot as plt
plt.imshow(img1)




# =============================================================================
# Pipeline 구축
# =============================================================================
# pipeline : 여러 모델을 결합하여 하나의 모델로 표현
# f(x)|g(x) > g(f(x))

from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.decomposition import PCA

pipe = make_pipeline(PCA(n_components=256), knn(n_neighbors=5))

pipe.fit(train_x, train_y)       # input data가 이미 PCA가 된 dataset 아님 주의
pipe.score(train_x, train_y)
pipe.score(test_x, test_y)

dir(pipe[0])                     # pipe 모델의 첫번째 모델이 갖는 메서드 목록
pipe[0].components_              # 유도된 가중치 shape(인공변수 개수, 총 row 개수)
pipe[0].components_.shape        # (256, 2914) (인공변수 개수, 총 row 개수)


