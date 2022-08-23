# -*- coding: utf-8 -*-
import pandas as pd
from pandas import Series
import numpy as np

# 1. student.csv, exam_01.csv 파일을 읽고
stud = pd.read_csv('data/student.csv', encoding = 'cp949')
exam = pd.read_csv('data/exam_01.csv', encoding = 'cp949')

# 1) 1,2학년 학생의 몸무게의 평균 출력
stud.loc[stud['GRADE'].isin([1, 2]), 'WEIGHT'].mean()

# 2) DEPTNO 컬럼을 생성하여 학과번호를 쓰되,
#    제2전공명이 없으면 제1전공번호를, 있으면 제2전공번호 입력
stud['DEPTNO'] = np.where(stud['DEPTNO2'].isnull(), stud['DEPTNO1'], 
                          stud['DEPTNO2']).astype('int')

# 3) ID 컬럼에 숫자 0을 포함하는 사람의 이름, ID, 학년 출력
'0' in stud['ID']                 # stud['ID']에 0 원소 유무 여부
'0' in 'wooya2702'                # 문자열에 0 포함 여부
f_find = lambda x : '0' in x
stud.loc[stud['ID'].map(f_find), ['NAME', 'ID', 'GRADE']]

# 4) exam_01의 각 학생의 시험 성적을 student에 TOTAL 컬럼으로 추가
f_total = lambda x : exam.loc[exam['STUDNO'] == x, 'TOTAL'].iloc[0]
# iloc[0]의 목적: 2차원(Series) > 1차원(scalar) 
stud['TOTAL'] = list(map(f_total, stud['STUDNO']))

# 5) 각 학년별로 시험성적이 가장 높은 사람 이름, 학년, 성적 출력
f_hscore = lambda x : stud.loc[stud['GRADE'] == x, 'TOTAL'].max()
stud['MAX_TOTAL'] = stud['GRADE'].map(f_hscore)
stud.loc[stud['TOTAL'] == stud['MAX_TOTAL'], ['NAME', 'GRADE', 'TOTAL']]



# 2. 'test3.txt' 파일을 읽고 
df2 = pd.read_csv('data/test3.txt', sep = '\t', header = None )
# header default: 1st row (True in R)

# 1) 다음과 같은 데이터 프레임 형태로 변경
# 	     20대	30대 40대 50대 60세이상
# 2000년	  7.5	3.6	 3.5  3.3	1.5
# 2001년	  7.4	3.2	 3.0  2.8	1.2
# 2002년	  6.6	2.9	 2.0  2.0   1.1
# ....................................
# 2011년	  7.4	3.4	 2.1  2.1	2.7
# 2012년	  7.5	3.0	 2.1  2.1	2.5
# 2013년	  7.9	3.0	 2.0  1.9	1.9

# map 함수 이용 
f_year = lambda x : str(x) + '년'
df2.index = list(map(f_year, np.arange(2000, 2014)))
# array + 'str' 불가 > 시리즈로 데이터 형태 변환
df2.index = Series(np.arange(2000, 2014)).astype('str') + '년'
# 컬럼 변경
df2.columns = Series(np.arange(20, 61, 10)).astype('str') + '대'
# column(index)의 일부 변경 - index(&columns) 일부 수정 불가(전체 수정 가능)
df2.columns[4] = '60세이상'        # Index does not support mutable operations
a1 = list(df2.columns)
a1[-1] = '60세이상'
df2.columns = a1
# 컬럼명 일부 수정을 위한 메서드
df2 = df2.rename({'60대' : '60세이상'}, axis = 1)

# 2) 2010년부터의 20~40대 실업률력만 추출하여 새로운 데이터프레임 생성
df3 = df2.iloc[df2.index.isin(df2.index[10:]), 0:3]
df3 = df2.loc['2010년': , '20대':'40대']

# 3) 30대 실업률을 추출하되, 소수점 둘째자리의 표현식으로 출력
df2['30대'].map(lambda x : '%.2f' % x)

# 4) 60세 이상 컬럼 제외
df2.drop('60대이상', axis = 1)




