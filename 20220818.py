# -*- coding: utf-8 -*-

run my_profile

# =============================================================================
# 데이터프레임
# =============================================================================
import pandas as pd
std = pd.read_csv('data/student.csv', encoding = 'cp949')
type(std)                        # 데이터 타입 확인
std.dtypes                       # 각 컬럼명, 컬럼별 데이터타입

std.columns                      # 컬럼명 확인
std.index                        # rowname 확인 

# [ Indexing ]
std.iloc[0, 0]                   # positional indexing(위치기반)
std.loc[0, 'GRADE']              # label indexing(이름 기반)
                                 # 0 : 위치값이자 rowname이므로 가능

std.index = std['STUDNO']        # rowname 변경
std




# =============================================================================
# working directory 변경
# =============================================================================
import os
os.getcwd()                      # 현재 작업 디렉토리 확인
os.chdir('경로')                  # 현재 작업 디렉토리 변경(현 세션에만 유효)

# Change Working directory in Spyder
# Tools > Preference > Current Working Directory




# =============================================================================
# 반복제어문
# =============================================================================
# 1. continue
#    킵특정 조건을 만날 때 하위 반복구문 스킵

for i in range(1, 101) :
    command 1                    # 100번 수행
    command 2                    # 100번 수행
    if i == 50 :
        continue
    command 3                    # 99번 수행
    

# 2. break
#    특정 조건을 만날 때 즉시 반복문 중단 
for i in range(1, 101) :
    command 1                    # 50번 수행
    command 2                    # 50번 수행
    if i == 50 :
        break
    command 3                    # 49번 수행 
    
    
# 3. exit 
#    특정 조건을 만날 때 즉시 프로그램 중단 
for i in range(1, 101) :
    command 1                    # 50번 수행
    command 2                    # 50번 수행
    if i == 50 :
        break
    command 3                    # 49번 수행
command 4                        # 수행 X


# 4. pass
#    if statement가 참일 때 return 대신 사용 가능
if a == 5 :
    pass
else :
    print('b')
    
    
    
# [ 연습 문제 ]
# 1 ~ 100 까지 짝수의 총합
vsum = 0
for i in range(1, 101) :
    if i % 2 == 1 :
        continue
    vsum = vsum + i
        
vsum = 0
for i in range(1, 101) :
    if i % 2 == 0:
        vsum = vsum + i

vsum = 0
for i in range(2, 101, 2) :
    vsum = vsum + i
    

# 연속 숫자 배열 만들기(in R: seq)
list(range(2, 101, 2)
np.arange(2, 101, 2)



# [ 연습 문제 ]
# 로또 번호 생성 프로그램(1 ~ 45, 6개 숫자)
import random

lonum = []
for i in range(1, 100) :
    vnum = random.randrange(1, 46)
    if vnum in lonum:
        continue
    lonum.append(vnum)
    if len(lonum) == 6 :
        break


lotto = []
while len(lotto) < 6 :
    vno = random.randrange(1, 46)
    if vno not in lotto :
        lotto.append(vno)
  
        
        

# =============================================================================
# Tuple 튜플
# =============================================================================
# 여러 상수를 전달하기 위한 목적으로 생성
# 1차원, 리스트와 유사
# Read Only, 수정 불가
# 나열 시 튜플로 생성됨

a2 = 1, 2, 3, 4                  # (1, 2, 3, 4)
type(a2)

a2[0] = 10                       # 삽입, 수정 불가
del(a2[0])                       # 불가
del(a2)                          # 가능 

t1 = (1, 2, 3)
t1 = 1, 2, 3
t1 = tuple([1, 2, 3])




# =============================================================================
# Dictionary 딕셔너리
# =============================================================================
# R의 리스트와 유사
# key-value 구조
# 2차원 X

# 1. 생성
d1 = {'name' : ['smith', 'allen'],
      'sal' : [800, 900]}
type(d1)


# 2. 색인
d1['name']
d1.get('name')


# 3. 변경
d1['comm'] = [100, 200]          # key 추가
d1['comm'] = None                # key 값 변경 > None
del(d1['comm'])                  # key 삭제


# 4. 활용
# 1) 데이터 프레임 생성
df1 = DataFrame({'ename' : ['a', 'b'], 'empno' : [1, 2]})

# 2) 대응관계 표현
d2 = {'a' : 100, 'b' : 110, 'c' : 120}
Series(['a', 'b', 'c']).map(d2)
Series(['a', 'b', 'c']).replace(d2)


# [ 참고 ] replace method
# 1. 기본 method: 벡터 연산 불가, 패턴 치환(문자열 일부 치환)
# 2. pandas method: 벡터 연산 가능, 값 치환(문자열 일부 치환 불가, 완전 일치 여부 중요)



 
# =============================================================================
# Set 세트
# =============================================================================
# 딕셔너리의 키 집합
# 중복 허용 불가

set(d1)                          # key 추출
s1 = set(['a', 'b', 'c', 'a'])   # 중복 제거

list(s1).append('d')             # 파생 객체를 이용한 원본 수정은 불가
l1 = list(s1)                    # 객체 선언 후 삽입
l1.append('d')




