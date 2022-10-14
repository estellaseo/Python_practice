# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import nan as NA
from pandas import Series, DataFrame

# =============================================================================
# 날짜 연산
# =============================================================================
from datetime import datetime

# 1. 현재 날짜 출력
d1 = datetime.now()

type(d1)                            # datetime.datetime 파이썬 제공 날짜 형식



# 2. 년, 월, 일 추출
d1.year                             # 년 추출
d1.month                            # 월 추출
d1.day                              # 일 추출



# 3. 형식 변환
#   (리턴 타입 : 문자) 
d1.strftime('%Y')                   # 문자열 년 추출
d1.strftime('%m')                   # 문자열 월 추출
d1.strftime('%d')                   # 문자열 일 추출
d1.strftime('%A')                   # 문자열 요일 추출



# 4. 날짜 파싱
d2 = '2022/06/21'
type(d2)
d2.strftime('%A')                   # 문자열 형태이므로 날짜 파싱 불가

#    1) datetime.strptime(date_string, format)
#       : 벡터 연산, 포맷 생략 불가 
#         > mapping or 반복문을 통해 처리
d2 = datetime.strptime(d2, '%Y/%m/%d')
d3 = ['2022/10/11', '2022/10/12', '2022/10/13', '2022/10/14']


#    2) pd.to_datetime
#       : 벡터 연산, 포맷 생략 가능(년월일 순일 때 자동 파싱)
d3 = pd.to_datetime(d3)
type(d3)                            # pandas.core.indexes.datetimes.DatetimeIndex


#    3) pd.read_csv의 parse_dates 옵션
#       : 년월일 순일 때 자동 파싱
emp = pd.read_csv('data/emp.csv')
emp['HIREDATE']                     # object

emp = pd.read_csv('data/emp.csv', parse_dates = 'HIREDATE')
emp['HIREDATE']                     # 날짜 타입(datetime64)


#    4) datetime
datetime(2022, 10, 14)


# [ 참고 ] strptime, strftime usage 출력
import time
time.strftime?



# 여러 객체 동시 fetch 방법
movie = pd.read_csv('data/movie_ex1.csv', encoding = 'cp949')

# 1) for + zip function
date = []
for i, j, z in zip(movie['년'], movie['월'], movie['일']):
    date.append(datetime(i, j, z))

# 2) map function
list(map(datetime, movie['년'], movie['월'], movie['일']))



# [ 참고 ] map 함수와 메서드의 차이
# 1. map 함수
#    - 1차원 객체 모두 사용, 여러 객체 동시 전달 가능
#    - ex) list(map(f1, s1, s2, s3))

# 2. map 메서드
#    - Series에만 사용 가능, 여러 객체 동시 전달 불가
#    - ex) s1.map(f1)



# 5. 연속적 날짜 출력(in R : seq)
np.arange(1, 101)

pd.data_range(start,                # 시작날짜(문자열가능)
              end,                  # 끝날짜(문자열가능)
              periods,              # 개수
              freq)                 # 단위(7일씩, 한 달씩)

pd.date_range('2022/01/01', '2022/01/31')
pd.date_range('2022/01/01', '2022/01/31', freq = '7D')
pd.date_range('2022/01/01', '2022/01/31', freq = 'W')    # 단위기간 매주 일요일
pd.date_range('2022/01/01', '2022/01/31', freq = 'W-MON')# 단위기간 매주 월요일

pd.date_range('2022/01/01', '2022/12/31', freq = 'M')    # 매 월 마지막 날짜 리턴
pd.date_range('2022/01/01', '2022/12/31', freq = 'BMS')  # 매 월 첫 영업일

# 시간대 표시, but 공휴일 적용 X
pd.date_range('2022/01/01', '2022/12/31', freq = 'BMS', tz = 'Asia/Seoul')

pd.date_range('2020/01/01', '2022/12/31', freq = 'Y')        # 매 년 마지막 날짜
pd.date_range('2022/01/01', '2022/12/31', freq = 'WOM-3FRI') # 매 월 셋째주 금요일



# [ 참고 ] 주식개장일 여부 확인 방법
# pip install exchange_calendars
import exchange_calendars as ec
XKRX = ec.get_calendar('XKRX')              # 한국기준 캘린더 다운
XKRX.is_session('2022/10/10')



# 6. 날짜 연산
d1 + 1                              # 날짜와 숫자의 연산 불가
                                    # 1에 대한 단위 기간(offset) 전달해야 연산 가능                                  
#    1) timedelta
#       일, 시, 분, 초, 주 단위 가능
from datetime import timedelta

timedelta(days=0, 
          seconds=0, 
          microseconds=0, 
          milliseconds=0, 
          minutes=0, 
          hours=0, 
          weeks=0)

d1 + timedelta(100)                 # 100일 뒤
d1 - timedelta(100)                 # 100일 전


#    2) offset
from pandas.tseries.offsets import Day, Hour, Second, MonthBegin, MonthEnd

d1 + Day(100)
d1 + MonthEnd(1)                    # 2022-10-14 기준 > 2022-10-31
d1 + MonthEnd(2)                    # 2022-10-14 기준 > 2022-11-30

d1 - MonthBegin(1)                  # 2022-10-14 기준 > 2022-10-01
d1 - MonthBegin(2)                  # 2022-10-14 기준 > 2022-09-01


#    3) relativedelta
from dateutil.relativedelta import relativedelta

d1 + relativedelta(days = 100)
d1 + relativedelta(months = 3)


#    4) pd.DateOffset
d1 + pd.DateOffset(days = 100)
d1 + pd.DateOffset(months = 3)



# 7. 날짜 - 날짜 연산
d1 - d2                        # (days=115, seconds=36870, microseconds=315824)
(d1 - d2).days



# 8 . 날짜 색인(날짜를 index로 가지고 있는 경우)
s1 = Series(np.arange(1, 101), index = pd.date_range('2022/01/01', periods=100))

s1['2022-01-01']
s1['2022-01']                  # 년-월까지만 전달해도 해당 월 날짜 색인 가능
s1['2022-01':'2022-03']        # 특정 구간의 날짜 색인 가능
s1['2022']                     # 특정 연도 색인 가능



# 9. 단위 기간에 대한 연산
#    - 일별 데이터 > 월별 데이터로 변경
#    - 날짜를 index로 가지고 있는 경우에만 사용
s2 = s1.resample('M').mean()   # 단위기간 조정(월별)
s1.resample('W').mean()        # 주별 평균

s2.resample('D').sum()
s2.resample('D').asfreq().fillna(method = 'bfill') / 30



s3 = Series(np.random.randint(1, 100, 12), 
            index = pd.date_range('2022/01/01', '2022/12/31', freq = 'M'))

s3['2022-01-01'] = NA                  # 1월 데이터도 일별 데이터로 만들기 위해 삽입
s3.resample('D').asfreq().bfill() / 30 # 월별 데이터 > 일별 데이터




# [ 연습 문제 ]
# card_history.csv 파일 데이터는 2022/01/01 기준 매주 일요일 작성된 가계부로 가정함
# 일별 데이터로 변환(일주일 지출 / 7 = 일별 지출)
card = pd.read_csv('data/card_history.csv', encoding = 'cp949')
card = card.drop('NUM', axis = 1)
card = card.applymap(lambda x : int(x.replace(',', '')))

card['날짜'] = pd.date_range('2022/01/01', freq = 'W-SUN', periods = len(card))
card = card.set_index('날짜')

card.resample('D').asfreq().bfill() / 7
 


