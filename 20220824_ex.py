# -*- coding: utf-8 -*-
run my_profile

# 1. kimchi_test.csv 파일을 읽고
df1 = pd.read_csv('data/kimchi_test.csv', encoding = 'cp949')

# 1) 제품별 수량과 판매금액의 총 합
df1.groupby(by = '제품')[['수량', '판매금액']].sum()

# 2) 판매월 별 수량의 총 합과 판매금액의 평균
d1 = {'수량' : 'sum', '판매금액' : 'mean'}
df1.groupby(by = '판매월')[['수량', '판매금액']].agg(d1)

# 3) 판매월 별 판매수량이 가장 많은 제품과 판매수량 동시 출력
df1_total = df1.groupby(by = ['판매월', '제품'], as_index = False)['수량'].sum()
df1_final = df1_total.groupby(by = '판매월', as_index = False)['수량'].max()
pd.merge(df1_total, df1_final, on = ['판매월', '수량'])

# 4) 판매연도별, 판매월별, 제품별 수량이 높은순서대로 정렬
df1_total2 = df1.groupby(by = ['판매년도', '판매월', '제품'], 
                         as_index = False)['수량'].sum()
df1_total2.sort_values(by = ['판매년도', '판매월', '수량'],
                      ascending = [True, True, False])


# 5) 김치별 월별 평균 판매량을 구한 뒤
#    김치별로 평균 판매량보다 낮은 판매량을 기록한 월 확인
df1_monsum = df1.groupby(by = ['제품', '판매월'], as_index = False)['수량'].sum()
df1_mean = df1_monsum.groupby(by = '제품', as_index = False)['수량'].mean()

df1_con = pd.merge(df1_monsum, df1_mean, on = '제품')
df1_con.loc[df1_con['수량_x'] < df1_con['수량_y'], ['제품', '판매월']]




# 2. crime.csv 파일을 읽고
df2 = pd.read_csv('data/crime.csv', encoding='cp949')

# 1) 검거/발생*100 값을 계산 후 rate 컬럼에 추가
df2['RATE'] = round(df2['검거'] / df2['발생'] * 100, 2)

# 2) 각 년도별 cctv수의 평균을 계산하여 각 평균값 보다 cctv수가 작은 행만 선택하여 출력
df2_mean = df2.groupby(by = '년도', as_index = False)['CCTV수'].mean()
df2_con = pd.merge(df2, df2_mean, on = '년도')
df2_con.loc[df2_con['CCTV수_x'] < df2_con['CCTV수_y'], :]




# 3. 한국소비자원.xlsx를 읽고 
df3 = pd.read_excel('data/한국소비자원.xlsx')

# 행의 수, 컬럼 수
df3.shape

# 특수문자 NA 처리
df3.dtypes                                         # 데이터 형태 우선 확인 필수
df3['판매가격'].astype('float')                     # 특수문자 여부 확인('.')
df3['판매가격'] = df3['판매가격'].replace('.', NA).astype('float')

# 1) 각 상품별, 판매처별 평균가격 출력
df3_mean = df3.groupby(by = ['상품명', '판매업소'], as_index = False)['판매가격'].mean()

# 2) 위 데이터를 사용하여 각 상품별로 가격이 가장 낮은 판매처 확인
# merge 이용
df3_con = df3_mean.groupby(by = '상품명', as_index = False)['판매가격'].min()
pd.merge(df3_mean, df3_con, on = ['상품명', '판매가격'])
# rank 이용
df3_mean.loc[df3_mean.groupby('상품명')['판매가격'].rank(method = 'min') == 1, :]

# [ 참고 ] indexing
# 색인으로 전달할 수 있는 객체의 형태 : 리스트, 1차원 배열, 시리즈
# 예시 ) 출력이 데이터프레임일 경우 컬럼 선택을 통해 차원 축소(시리즈)


