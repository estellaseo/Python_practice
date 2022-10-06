# -*- coding: utf-8 -*-
run my_profile

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus = False)

# =============================================================================
# 시각화
# =============================================================================
# [ 시각화 모드 전환 방법 ] 
# 1. spyder
Tools > Preferences > IPython console > Graphics > Graphics backend에서
backend를 Automatic으로 변경

# 2. ipython console
# 2-1) 실행 시 시각화모드 전환
ipython --pylab

# 2-2) 이미 ipython 실행한 경우 시각화모드 전환
%matplotlib qt




# =============================================================================
# plot
# =============================================================================
# 1. Series에서의 plot
#    - index가 x축 눈금으로 자동 전달됨
#    - index의 이름이 x축 이름으로 자동 전달됨

card = pd.read_csv('card_history.csv', encoding='cp949')
card = card.set_index('NUM')

# index 이름 > x축 이름 자동 전달
card.index.name = 'Date'

card = card.applymap(lambda x : int(x.replace(',','')))
card['식료품'].plot()


# 2. DataFrame에서의 plot
#    - wide data일 때 각 컬럼마다 서로 다른 선 그래프 출력
#    - 자동으로 legend 표현(best position)
#    - column의 이름이 legend 이름으로 자동 전달
card.columns.name = '지출항목'
card.plot()




# =============================================================================
# figure, subplot
# =============================================================================
# figure : 그래프를 출력하는 화면(도화지)
# subplot : 도화지 내 실제 그래프가 그려지는 공간(하나의 화면에 여러 공간 할당 가능)
# 일반적으로 그래프를 그리면 하나의 figure, 하나의 subplot 할당


# figure, subplot 직접 할당
# 1) 개별 전달
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,         # 행의 수
                       2,         # 컬럼 수
                       1)         # subplot 순번

ax2 = fig1.add_subplot(2, 2, 2)
ax3 = fig1.add_subplot(2, 2, 3)
ax4 = fig1.add_subplot(2, 2, 4)

ax3.plot(card['식료품'])


# 2) 한꺼번에 전달
plt.subplots(nrows = ,            # 행의 수
             ncols = ,            # 컬럼 수
             sharex = False,      # x축 눈금 고정 여부
             sharey = False)      # y축 눈금 고정 여부

fig, ax = plt.subplots(2,2)
ax[0,0].plot(card['식료품'])




# =============================================================================
# plot 옵션
# =============================================================================
plt.plot()
card.plot(data,                   # 그래프를 표현할 데이터(Series, DataFrame)
          x,                      # x축 좌표 
          y,                      # y축 좌표
          kind = 'line',          # 그래프 종류('bar', 'hist', 'kde', 'box', 'pie', 'scatter')
          ax,                     # subplot 이름
          subplots = False,       # 매 컬럼마다 서로 다른 subplot에 시각화 할 지 여부
          sharex = ,
          sharey = ,
          title = ,               # 그래프 메인 제목
          legend = ,              # 범례(default = True)
          style,                  # 선 스타일('go--')
          marker = ,              # 점 모양
          linestyle = ,           # 선 모양
          color = ,               # 색
          xlim = ,                # x축 범위
          ylim = ,                # y축 범위
          xlabel = ,              # x축 이름
          ylabel = ,              # y축 이름
          xticks = ,              # x축 눈금
          yticks = ,              # y축 눈금
          rot)                    # 회전 각도(xticks, yticks) 
          

card['식료품'].plot(ylim = [0,50000], style='b^--',
                    title='식료품 지출 현황',
                    ylabel = '금액',
                    xticks = card.index, rot = 45)          




# =============================================================================
# 옵션 별도 설정 방법(상세 옵션 전달 가능)          
# =============================================================================
plt.title('zzzzzz')
plt.ylabel('aaa', loc = 'top', rotation = 0)
plt.ylim([0,40000])

dir(plt)                          # plt에서 호출 가능한 메서드 목록

plt.xticks(ticks=,                # 각 축 눈금
           labels=,               # 각 축 이름
           rotation=30)

plt.tick_params(axis='x',         # 눈금 설정 위치(x축 눈금 조절, y축 눈금 조절 여부)
                direction='in',   # 눈금 표현 방향
                color='r',        # 눈금 색
                width=2,          # 눈금 두께
                length=3,         # 눈금 길이
                pad=6,            # 눈금과 레이블과(눈금이름)의 거리
                labelsize=14,     # 눈금이름 크기
                labelcolor)       # 눈금이름 색

plt.ylim([0,100000])

plt.ylabel('주문건수',             # 축 제목
           rotation=0,            # 회전 각도
           x = 9,                 # x축 위치
           y = 1.05,              # y축 위치
           loc = ,                # 종합 위치(top, right, left, center, ...)
           labelpad=15,           # 여백
           fontdict=)             # 폰트 설정
           
plt.title('요일별 업종 주문건수 비교', 
          fontsize=20, 
          c='r', 
          weight='bold', 
          y = 1.05,
          fontdict=)

plt.legend(fontsize=8,            # 글자크기
           fontdict=,             # 폰트 설정
           loc = 'best'           # 위치(best, lower left, upper right, ...)
           ncol = 2               # 범례 출력시 2차원 형태로 출력 가능
           title='진료과목',       # 범례 제목
           facecolor=,            # 색 채우기
           edgecolor=,            # 테두리 색
           labelspacing=1,        # 범례 항목 간 위아래 간격 
           borderpad=4,           # 범례전체 상하좌우 여백
           frameon=True,          # 범례 테두리 표시 여부
           shadow=True)           # 그림자 출력 여부




# [ 연습 문제 ] 
# cctv.csv 파일을 읽고 구별 검거율이 가장 높은 top5 구에 대해 각 연도별 검거율 
# 증감추이를 구별로 비교할 수 있도록 선 그래프 시각화
cctv = pd.read_csv('data/cctv.csv', encoding = 'cp949')

# step 1) 검거율 컬럼 생성
cctv['검거율'] = round(cctv['검거'] / cctv['발생'] * 100, 2)

# step 2) 구별 검거율 top5 선택
ind1 = cctv.groupby('구')['검거율'].mean().sort_values(ascending = False)[:5].index
cctv2 = cctv.loc[cctv['구'].isin(ind1), :]

# step 3) wide data 변경
cctv3 = cctv2.pivot_table(index = '년도', columns = '구', values = '검거율')

# step 4)시각화
cctv3.plot()

# 옵션 전달
plt.ylim([0, 120])
plt.ylabel('검거율', rotation = 0)
plt.title('구별 검거율 변화', fontdict = font1)
plt.xticks(cctv3.index)
plt.legend(fontsize = 9, title = '구이름', ncol = 2)

# fontdict(폰트 설정에 대한 딕셔너리)
font1 = {'family' : 'Malgun Gothic',
         'color' : 'forestgreen',
         'style' : 'italic',
         'size' : 14}




# =============================================================================
# bar plot
# =============================================================================
# 각 행별로 컬럼에 대한 값을 비교하기위한 막대그래프 출력

kimchi = pd.read_csv('data/kimchi_test.csv', encoding = 'cp949')

kimchi2 = kimchi.pivot_table(index='판매월', columns='제품', values='수량', aggfunc='sum')

kimchi2.plot(kind='bar')
plt.ylim([0,300000])
plt.xticks(rotation=0)
plt.legend(fontsize = 9,        # 본문 글자크기
           title='김치종류',
           title_fontsize=11,   # title 글자크기
           labelspacing = 1,    # 범례 항목간 위아래 간격
           borderpad = 2)       # 범례 전체 상하좌우 여백

plt.barplot()




# =============================================================================
# pie chart
# =============================================================================
plt.pie(x,                      # 각 파이를 표현할 숫자
        explode = ,             # 중심에서 벗어나는 정도 설정(파이별 서로 다른값 전달 가능)
        labels =,               # 각 파이 이름
        colors = ,
        autopct = ,             # 값의 표현 포맷('%.2f')
        startangle = ,          # 시작 위치
        radius,                 # 파이 크기
        counterclock=,          # 시계방향 진행여부(default=True)
        wedgeprops=,            # 부채꼴 모양 설정
        shadow=)                # 그림자 출력여부(default=False)


# pie 시각화
# 색상표 : https://wepplication.github.io/tools/colorPicker/
labels = ['apple','banana','melon','mango']
colors = ['#d96353', '#53d98b', '#53a1d9', '#fab7fa']   
explodes = [0.1,0.1,0.1,0.5]
wedgeprops = {'width':0.7, 'edgecolor':'#d395d0', 'linewidth':5}

plt.pie([10,12,14,20], labels=labels, colors = colors, 
        explode = explodes, wedgeprops = wedgeprops)




# [ 연습 문제 ]
# card_history.csv 파일을 읽고
# 1. 요일별(2022년 9월 데이터로 가정) 각 지출품목에 대한 지출 총 합을 비교하는 막대그래프 시각화
card = pd.read_csv('data/card_history.csv', encoding='cp949')
card = card.set_index('NUM')
card = card.applymap(lambda x : int(x.replace(',','')))

# step 1) 날짜 문자열 결합
date1 = '2022/09/' + card.index.astype('str')

# step 2) 날짜 파싱 및 요일 출력
from datetime import datetime
date2 = date1.map(lambda x : datetime.strptime(x, '%Y/%m/%d').strftime('%A'))

# step 3) 그룹 연산
card2 = card.groupby(date2).sum().iloc[[1, -2, -1, -3, 0, 2, 3]]

# step 4) 시각화
card2.plot(kind = 'bar')


# 2). 각 지출품목에 대한 전체 지출 총 합을 비교하는 파이차트 시각화
s1 = card.sum()
plt.pie(s1, labels = s1.index)




# =============================================================================
# Scatter
# =============================================================================

# 예제) iris data 산점도 출력
from sklearn.datasets import load_iris
iris = load_iris()

iris_x = iris['data']
iris_y = iris['target']

xnames = iris['feature_names']

plt.subplot(2, 2, 1)
plt.spring()
plt.scatter(iris_x[:, 0], iris_x[:, 1], c = iris_x[:, 1])
plt.xlabel(xnames[0])
plt.ylabel(xnames[1])
plt.colorbar()


plt.subplot(2, 2, 2)
plt.summer()
plt.scatter(iris_x[:, 1], iris_x[:, 2], c = iris_x[:, 2])
plt.xlabel(xnames[1])
plt.ylabel(xnames[2])
plt.colorbar()


plt.subplot(2, 2, 3)
plt.autumn()
plt.scatter(iris_x[:, 3], iris_x[:, 0], c = iris_x[:, 0])
plt.xlabel(xnames[3])
plt.ylabel(xnames[0])
plt.colorbar()


plt.subplot(2, 2, 4)
plt.winter()
plt.scatter(iris_x[:, 3], iris_x[:, 1], c = iris_x[:, 1])
plt.xlabel(xnames[3])
plt.ylabel(xnames[1])
plt.colorbar()


# matplotlib에서 제공하는 colormap 확인
from matplotlib import cm
plt.colormaps()                             # Greys, Reds, Pastel1, etc.




# =============================================================================
# hist
# =============================================================================
s1 = Series(np.random.randn(1000))

s1.hist(bins = 40)
plt.grid(True)                              # 격자 설정
plt.grid(False)                             # 격자 설정 해제

# 격자 설정 옵션
plt.grid(True, axis = 'y', color = 'r', linestyle = '--', linewidth = 0.8)

plt.hist(s1, bins = 40, density = False)    # y축 값 : 도수
plt.hist(s1, bins = 40, density = True)     # x축 값 : 확률

plt.rc('axes', unicode_minus = False)
s1.plot(kind = 'kde')




# =============================================================================
# boxplot
# =============================================================================
plt.boxplot(iris_x)
plt.xticks(ticks = [1, 2, 3, 4],            # 눈금
           labels = xnames)                 # 눈금에 대한 이름




# =============================================================================
# 텍스트 삽입
# =============================================================================
plt.text(x,                                 # x축 좌표
         y,                                 # y축 좌표 
         s,                                 # text
         fontdict, 
         rotation, 
         bbox = )                           # 텍스트 박스 설정

plt.text(2, 6, 'aa')


# bbox dict
box1 = {'boxstyle' : 'square',              # squre, round 
        'ec' : 'red',                       # edge color
        'fc' :'b',                          # face col=/or
        'linestyle' : '--',
        'linewidth' : 2}

plt.text(2, 6, 'aa', bbox = box1) 




# =============================================================================
# 그래프 채우기
# =============================================================================
x = [1, 2, 3, 4]
y = [2, 3, 5, 10]

plt.plot(x, y)
plt.fill_between(x[1:3],                    # x축 좌표  
                 y[1:3],                    # y축 좌표
                 y2 = 0,                    # 가장 밑단 y값
                 alpha = 0.5,               # 투명도  
                 color = 'lightgray')       # 색상




# =============================================================================
# 수직선, 수평선 그리기
# =============================================================================
s1 = Series(np.random.randn(1000))
s1.plot(kind = 'kde')

# 수평선 그리기 1) 상대적 기준(범위 0~1)
plt.axhline(y = 0,                          # y축 좌표
            xmin = 0.2,                     # 가장 작은 시작값(0)
            xmax = 0.8)                     # 가장 큰 끝값(1)

# 수평선 그리기 2) 실제 좌표 기반
plt.hlines(y=0, 
           xmin = -2,
           xmax = 2)

# 수직선 그리기 1) 상대적 기준
plt.axvline(x = 0,                          # y축 좌표
            ymin = 0.2,                     # 가장 작은 시작값(0)
            ymax = 0.8)                     # 가장 큰 끝값(1)

# 수직선 그리기 2) 실제 좌표 기반
plt.vlines(x = 0, ymin = 0, ymax = 0.4)



