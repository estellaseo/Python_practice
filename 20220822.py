# -*- coding: utf-8 -*-

# =============================================================================
# pandas
# =============================================================================
# Series, DataFrame 자료 구조 제공
# 반복 연산, 치환 및 삭제 등의 기능 편리
import pandas as pd
import numpy as np




# =============================================================================
# Series 시리즈
# =============================================================================
# 1차원
# 서로 다른 데이터 타입을 허용하지 않음
# 대부분의 벡터 연산 지원(산술연산, 조건전달, ... )


# 1. 생성
from pandas import Series
s1 = Series([1, 2, 3, 4, 5])
s2 = Series([1, 2, 3, 4, '5'])

s1.dtype                                 # dtype('int64')
s2.dtype                                 # dtype('O')

s3 = Series([1, 2, 3, 4, 5], index = ['a', 'b', 'c', 'd', 'e'])
s3                                       # 생성 시: index 전달
# 이미 index를 가지고 있는 경우 - index를 이용한 추출
Series(s3, index = ['A', 'B', 'C', 'D', 'E']) 

# index 변경 메서드
s3.index = ['A', 'B', 'C', 'D', 'E']
s3.reindex(['C', 'D', 'E', 'A', 'B'])
s3.reindex(['A', 'B', 'O'])              # error X - O NaN
s3.reindex(['A', 'B', 'O'], 
           fill_value = 0)               # NA 치환값(상수 값)
s3.reindex(['A', 'B', 'O', 'C', 'F'],
           method = 'ffill')             # NA 치환값(이전 값)



# 2. 색인
s3[0]                                    # positional indexing
s3[0:2]                                  # slice indexing
s3[['A', 'B']]                           # label indexing
s3[s3 > 3]                                 #h
s3[['A', 'B', 'O']]                      # error - "['O'] not in index"


# 3. 연산
s1 + s3                                  # 같은 index를 갖는 값끼리 연산
s4 = Series([1, 2, 3, 4, 5, 6]) 
s1+ s4                                   # index가 5인 경우 NA return
                                         # 5 index가 없을 경우 NaN

s1.reindex([0, 1, 2, 3, 4, 5]) + s4      # 양쪽 index를 일치시킨 후 연산
s1.reindex([0, 1, 2, 3, 4, 5]).fillna(0) + s4 


# 4. 산술연산(+, -, *, /) 메서드
# key가 서로 일치하지 않는 경우 NA 수정 후 리턴

s1.add(s4, fill_value = 0)               # 더하기
s1.sub(s4, fill_value = 0)               # 빼기
s1.mul(s4, fill_value = 1)               # 곱하기
s1.div(s4, fill_value = 1)               # 나누기


# 5. 수학통계 메서드
s1.sum()
s1.mean()

s3.argmin()                              # 최소값 위치 리턴
s3.idxmin()                              # 최소값 key 리턴




# =============================================================================
# DataFrame 데이터프레임
# =============================================================================
# 2차원
# 같은 컬럼 내 하나의 데이터 타입 허용(시리즈)
# key-value 구조


# 1. 생성
from pandas import DataFrame
a1 = np.arange(1, 13).reshape(4, 3)
df1 = DataFrame(a1)

# 생성 시 index, column 전달 가능
df2 = DataFrame(np.arange(10, 121, 10).reshape(4, 3),
                index = [1, 2, 3, 4], columns = list('abc'))
df3 = DataFrame({'col1' : [1, 2, 3, 4], 'col2' : ['a', 'b', 'c', 'd']})


# 2. 기본 메서드
df3.dtypes
df3.index
df3.columns
df3.values                               # key 제외 값만 출력(array 리턴)


# 3. 색인
#    - iloc : 위치값만 전달 가능
#    - loc : 이름, 논리값 전달

df3.col1                                 # key indexing
df3['col1']

df3.iloc[:, 0]                           # 위치를 이용한 색인
df3.loc[:, 'col1']                       # 이름을 이용한 색인

df3.iloc[0:3, :]                         # slice 색인 가능
df3.loc[:, 'col1':'col2']                # 문자 색인 시 마지막 범위 포함하여 출력

df3.loc[[0,3], ['col1', 'col2']]         # 리스트를 이용한 색인 가능



# [ 연습 문제 ]
stud = pd.read_csv('data/student.csv', encoding = 'cp949')

# 1) 1학년 키 평균
stud.loc[stud['GRADE'] == 1, 'HEIGHT'].mean()

# 2) 이윤나와 김주현의 이름, 학년, 제1전공번호 출력
stud.loc[(stud['NAME'] == '이윤나') | (stud['NAME'] == '김주현'),
         ['NAME', 'GRADE', 'DEPTNO1']]

# [ 참고 ] in 연산자 사용
stud.loc[stud['NAME'].isin(['이윤나', '김주현']), ['NAME', 'GRADE', 'DEPTNO1']]




# =============================================================================
# replace 메서드
# =============================================================================
# 1. 기본 메서드(문자열 메서드)
#    - 문자열에만 적용 가능
#    - 문자열 치환: 문자열의 일부 삭제 및 치환 가능
#    - 문자열 이외의 값으로 수정 불가

'abcde'.replace('a', 'A')                # 문자열 일부 치환 가능
'abcde'.replace('a', 1)                  # 숫자로 치환 불가


# 2. pandas 메서드
#    - pandas 객체(Series, DataFrame)에만 적용 가능 
#    - 값 치환 : 문자열의 일부 삭제 및 치환 불가, 값 치환 목적 사용
#    - 다른 데이터 타입, NA로의 치환 가능
#    - 정규표현식 전달 가능

Series('abcde').replace('a', 'A')        # 문자열 일부 치환 불가

s1 = Series([1, 2, 3, NA])
s1.replace(NA, 0)                        # NA값 치환 가능

s2 = Series(list('abcde'))
s2.replace('a', 1)                       # 다른 데이터타입으로 치환 가능
s2.replace('a', NA)                      # NA로 치환 가능




# =============================================================================
# NA 치환
# =============================================================================
# 방법 1)
s1[pd.isnull(s1)] = 0
# 방법 2)
np.where(pd.isnull(s1), 0, s1)
# 방법 3)
s1.fillna(0)
# 방법 4)
s1.replace(NA, 0)




# =============================================================================
# index, column 변경
# =============================================================================
df1 = DataFrame(np.arange(1, 17).reshape(4, 4))
df1.index = list('abcd')
df1.columns = list('abcd'.upper())

df1 = df1.set_index('A')                 # 본문에 있는 컬럼을 지정하여 index로 설정

df1.index.name                           # index 이름 전달
df.columns.name                          # column 이름 전달




# =============================================================================
# drop 메서드
# =============================================================================
# - pandas 객체에 사용
# - 원하는 행, 컬럼 제외 가능
df1
df1.drop('B',                            # 제외할 행 또는 컬럼명
         axis = 1)                       # 방향(default: axis = 0)

df1.drop(5)                              # 숫자 전달 가능(행 이름)



# [ 연습 문제 ]
# apply.csv 파일을 읽고
df2 = pd.read_csv('data/apply.csv', encoding = 'cp949')

# 1) date 컬럼 추가: 2019/05/01 형태로 변경
f_plus = lambda x, y, z : str(x) + '/' + '%02d' %y + '/' + '%02d' %z
df2['date'] = list(map(f_plus, df2.iloc[:, 0], df2.iloc[:, 1], df2.iloc[:, 2]))


# [ 참고 ] pad 함수
'5'.ljust(2, '0')
'5'.rjust(2, '0')

# 2) id 컬럼과 passwd 컬럼 분리
f_id = lambda x : x.split('/')[0]
f_pw = lambda x : x.split('/')[1] if '/' in x else NA

df2['id'] = list(map(f_id, df2['id/passwd']))
df2['pw'] = list(map(f_pw, df2['id/passwd']))

df2.drop('id/passwd', axis = 1)

# [ 참고 ] np.where 사용 시 주의
np.where(con1, True, False)
# 조건1, 객체1, 객체2 정상 출력 및 모두 개수 일치해야 함



# [ 연습 문제 ]
# student.csv 파일과 exam_01.csv 파일을 읽고 각 학생의 시험 성적을 TOTAL 컬럼으로 추가
stud = pd.read_csv('data/student.csv', encoding = 'cp949')
exam = pd.read_csv('data/exam_01.csv', encoding = 'cp949')

f_total = lambda x : exam.loc[exam['STUDNO'] == x, 'TOTAL'].iloc[0]
stud['TOTAL'] = list(map(f_total, stud['STUDNO']))
