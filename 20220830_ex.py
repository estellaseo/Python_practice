# -*- coding: utf-8 -*-
run my_profile
from datetime import datetime

# 1. movie_ex1.csv 파일을 읽고
# 요일별 영화 이용비율이 가장 높은 연령대 확인
movie = pd.read_csv('data/movie_ex1.csv', encoding='cp949')

# 날짜/요일 컬럼 생성
d1 = movie['년'].astype('str')+'/'+movie['월'].astype('str')+'/'+movie['일'].astype('str')
movie['요일'] = d1.map(lambda x : datetime.strptime(x, '%Y/%m/%d').strftime('%A'))
# 불필요 컬럼 삭제
movie = movie.drop(['년', '월', '일', '지역-시도', '지역-시군구', '지역-읍면동'], axis = 1)
# 그룹핑
movie_sum = movie.groupby(by = ['요일', '연령대'])['이용_비율(%)'].sum()
movie_sum[movie_sum.groupby('요일').idxmax()].iloc[[1, -2, -1, -3, 0, 2, 3]]
# 주의 : 데이터프레임을 이용한 색인 불가 
#       > 리턴 데이터 타입이 데이터프레임일 경우 색인을 통한 이용비율 컬럼 출력 불가
#       > 컬럼 선택을 통한 차원 축소 후 색인 전달




# 2. multi_index_ex2.csv 파일을 읽고
multi = pd.read_csv('data/multi_index_ex2.csv', encoding = 'cp949')
# 1_A => 1월의 A지점의 의미로 해석!

# 1) 멀티인덱스, 멀티 컬럼 설정
# 멀티 인덱스
c1 = multi.iloc[:, 0].map(lambda x : str(x)[:1])    # 문자열 변환 후 색인
c2 = multi.iloc[:, 0].map(lambda x : str(x)[-1])
multi.index = [c1, c2]
multi = multi.drop('Unnamed: 0', axis = 1)
multi.index.names = [None, None]

# 멀티 컬럼
s1 = Series(multi.columns)
s1 = s1.map(lambda x : NA if ' ' in x else x).fillna(method = 'ffill')
multi.columns = [s1, multi.iloc[0, :]]
multi = multi.drop('n', level = 0, axis = 0)
multi.columns.names = [None, None]
# [ 참고 ] series.replace(정규 표현식 이용)
Series(multi.columns).replace(' ', NA, regex = True).ffill()

# 2) 컬럼을 요일별로 먼저 정리
multi.sort_index(axis = 1, level = [1, 0]).swaplevel(0, 1, axis = 1).loc[:, ['월', '화', '수', '목', '금']]

# 3) 지점별 지역별 매출 총 합 출력
multi = multi.replace(['.', '-', '?'], NA).astype('float')

multi.sum(axis = 0, level = 1).sum(axis = 1, level = 0)
multi.groupby(axis = 0, level = 1).sum().groupby(axis = 1, level = 0).sum()




# 3. employment.csv 파일을 읽고
emp = pd.read_csv('data/employment.csv', encoding = 'cp949')

# 인덱스 설정
emp = emp.set_index('고용형태')
# 멀티 컬럼 설정
emp.columns = [emp.columns.map(lambda x : x[:4]), 
               emp.iloc[0, :].map(lambda x : x.split(' ')[0])]
emp = emp.drop('고용형태', axis = 0)

# 1) 총근로일수 추출
emp.xs('총근로일수', axis = 1, level = 1)

# 2) 전체 컬럼에 대해 평균을 출력(총근로일수 평균(연도 상관없이))
# 결측치 처리
emp = emp.replace(['-', ''], NA)
# 데이터타입 float으로 변경
# NA가 있는 경우 replace 함수 이용 불가 > 조건문을 통해 NA가 아닐 때만 변환
emp = emp.applymap(lambda x : x if pd.isnull(x) else float(x.replace(',', '')))
emp.groupby(axis = 1, level = 1).mean()['총근로일수']


