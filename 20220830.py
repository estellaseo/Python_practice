# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 집합 연산자(1차원 객체)
# =============================================================================

# 1) index object

# 합집합
d1 = s1.index
d2 = s2.index

dir(d1)                                 # index object에 사용 가능한 method 목록

d1.union(d2)                            # 합집합
d1.intersection(d2)                     # 교집합
d1.difference(d2)                       # 차집합


# 예시) key가 다른 시리즈끼리 연산
s1 = Series([1, 2, 3, 4, 5], index = list('abcde'))
s2 = Series([1, 2, 3, 4, 5, 6], index = list('abcdef'))

s1 + s2                                 # key가 일치하지 않는 경우 NA 리턴
s1.add(s2, fill_value = 0)
s1.reindex(s2.index).fillna(0) + s2

s1 = Series([1, 2, 3, 4, 5], index = list('acefh'))
s2 = Series([1, 2, 3, 4, 5, 6], index = list('acdefg'))

# key가 일치하지 않는 경우 NA 출력 방지
s1.add(s2, fill_value = 0)
# add 메서드의 실행 원리
d3 = s1.index.union(s2.index)
s1.reindex(d3).fillna(0) + s2.reindex(d3).fillna(0)


# 2) numpy 집합 연산자
np.union1d(d1, d2)                      # 합집합
np.intersect1d(d1, d2)                  # 교집합
np.                                     # 차집합




# =============================================================================
# 데이터 합치기(데이터프레임 집합연산자)
# =============================================================================
emp = pd.read_csv('data/emp.csv')

emp_10 = emp.loc[emp['DEPTNO'] == 10, :]
emp_20 = emp.loc[emp['DEPTNO'] == 20, :]
emp_1020 = emp.loc[emp['DEPTNO'].isin([10, 20]), :]

# [ pd.concat ]
# 가로 방향 결합
pd.concat([emp_10, emp_20], axis = 0)   # 데이터 합치기(axis = 0 행 결합) : 합집합
pd.concat([emp_10, emp_20], axis = 0, ignore_index = True)

pd.concat([emp_10, emp_1020])           # UNION ALL과 유사한 합집합 
pd.concat([emp_10, emp_1020]).drop_duplicates()     # UNION 합집합

# 세로 방향 결합(merge or concat(axis = 1))
# concat 주의: index 일치한 행끼리 결합
emp_1 = emp[['EMPNO', 'ENAME', 'SAL']]
emp_2 = emp[['EMPNO', 'DEPTNO', 'COMM']]
emp_3 = emp_2.sort_values('DEPTNO', ignore_index = True)

pd.concat([emp_1, emp_2], axis = 1)     # 모든 컬럼 나열됨
pd.concat([emp_1, emp_3], axis = 1)     # index 기준으로 결합 > 잘못된 데이터

pd.merge(emp_1, emp_3, on = 'EMPNO')    # merge - best option


# 교집합
pd.merge(emp_10, emp_1020)              # natural join > 교집합


# 차집합
# (keep = False) > 중복된 값을 모두 제거
pd.concat([emp_1020, emp_10]).drop_duplicates(keep = False)



# [ 연습 문제 ]
# emp_1.xlsx, emp_2.xlsx, emp_3.xlsx 파일을 모두 합치기
emp_1 = pd.read_excel('data/emp_1.xlsx')
emp_2 = pd.read_excel('data/emp_2.xlsx')
emp_3 = pd.read_excel('data/emp_3.xlsx')

pd.concat([pd.merge(emp_1, emp_2, on = 'EMPNO'), emp_3], ignore_index = True)




# =============================================================================
# 날짜 변환(날짜 파싱/포맷 변경)
# =============================================================================

# [ 참고 ] 언어별 형 변환 함수 비교
# SQL    : to_date, to_char
# R      : as.Date, as.character
# Python : int, float, str > 형변환 함수로 날짜 파싱 및 포맷 변경 불가


# 1. strptime
#    날짜가 아닌 문자열에 대해 날짜로 인식
#    datetime 모듈 호출 시 사용
#    벡터 연산 불가

from datetime import datetime
datetime.strptime(date_string, format)

s1 = '2022/08/30'
d1 = datetime.strptime(s1, '%Y/%m/%d')
type(d1)                                # datetime.datetime(년, 월, 일, 시, 분)

d1 + 100                                # 서로 다른 데이터타입 연산 불가(날짜 + 숫자)

s2 = Series(['2022/01/01', '2022/01/02'])
datetime.strptime(s2, '%Y/%m/%d')       # 벡터연산 불가
d2 = s2.map(lambda x : datetime.strptime(x, '%Y/%m/%d'))


# 2. strftime
#    날짜의 포맷 변경(문자열 리턴)
#    datetime 모듈 호출 시 사용
#    벡터 연산 불가 

datetime.strftime(self, fmt)

# 함수 형식
datetime.strftime(d1, '%A')             # scalar에 대한 포맷 변경
# 메서드 형식
d1.strftime('%A')

datetime.strftime(d2, '%A')             # 벡터연산 불가
d2.map(lambda x : datetime.strftime(x, '%A'))
d2.map(lambda x : x.strftime('%A'))



# [ 연습 문제 ]
# emp.csv 파일을 읽고 입사일의 요일별 급여의 평균을 구하라
emp = pd.read_csv('data/emp.csv')

hiredate = emp['HIREDATE'].map(lambda x : datetime.strptime(x, 
                                                            '%Y-%m-%d %H:%M'))
emp['HIREDAY'] = hiredate.map(lambda x : datetime.strftime(x, '%A'))
round(emp.groupby(by = 'HIREDAY')['SAL'].mean(), 2).iloc[[1, -2, -1, -3, 0, 2]]

# data import에서 parse_dates 지정(년월일 형식만 자동 파싱 가능)
emp2 = pd.read_csv('data/emp.csv', parse_dates = ['HIREDATE'])




