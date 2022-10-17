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
people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)

people_x = people['data']
people_y = people['target']

people_x.shape                        # (3023, 5655) 3023개 이미지, 평탄화(87X65)
people_y                              # 각 인물의 이름 > 숫자 변환(0 ~ 61)
people['target_names']                # 각 인물의 실제 이름값

people_x[0, :]                        # RGB 변환된 값(0~255)
people_img = people['images']         # 평탄화 시키기 이전 데이터

plt.imshow(people_img[0, :, :])       # 첫번째 이미지 출력



# 2. 데이터 분리
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(people_x, people_y, random_state=0)



# 3. 모델링
#    1)KNN으로 분류
from sklearn.neighbors import KNeighborsClassifier as knn_c
m_knn = knn_c()
m_knn.fit(train_x, train_y)
m_knn.score(train_x, train_y)         # 0.4503
m_knn.score(test_x, test_y)           # 0.1825


#    2) PCA + knn
# 스케일링
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
people_x_sc = m_sc.fit_transform(people_x)


# PCA
from sklearn.decomposition import PCA
m_pca = PCA(5)
m_pca.fit(people_x_sc)


# knn
train_x, test_x, train_y, test_y = train_test_split(people_x_sc, people_y, random_state=0)
m_knn = knn_c()
m_knn.fit(train_x, train_y)
m_knn.score(train_x, train_y)         # 0.5050
m_knn.score(test_x, test_y)           # 0.2328



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



# [ 참고 ] 이미지 변환(RGB값 추출)
import imageio
img1 = imageio.imread('joshua.jfif')
img1.shape

import matplotlib.pyplot as plt
plt.imshow(img1)



