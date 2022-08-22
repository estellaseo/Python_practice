# -*- coding: utf-8 -*-

# =============================================================================
# 논리연산자
# =============================================================================
v1= 10

(v1 >= 3) and (v1 < 15)
(v1 >= 3) & (v1 < 15)

(v1 <= 3) or (v1 > 15)
(v1 <= 3) | (v1 > 15)

# [ 참고 ] 벡터 논리 연산자 - 여러개의 논리값 처리
l1 = [1, 2, 5, 10, 12, 17]
l1 >= 3                                  # 비교연산자 사용 불가

from pandas import Series
Series(l1) >= 3                          # 각 원소별 비교 가능
(Series(l1) >= 3) and (Series(l1) < 15)  # 에러 발생(여러 값 수행 X)
(Series(l1) <= 3) or (Series(l1) > 15)

(Series(l1) >= 3) & (Series(l1) < 15)    # 사용 가능
(Series(l1) <= 3) | (Series(l1) > 15)




# =============================================================================
# 벡터 연산
# =============================================================================
l2 = ['acd', 'dfdf', 'dfdf']
l2.upper()                               # 리스트에 적용 불가
Series(l2).upper()                       # Series에 적용 불가
l2[0].upper()                            # 문자열 적용 가능




# =============================================================================
# 사용자 정의 함수
# =============================================================================
# 1) 축약형
#    간단한 함수 정의에 용이 > 단순 리턴의 경우
function_name = lambda input_value : return_value
#    제한적 if문 지원 > else 생략 불가, elif 사용 불가
function_name = lambda input_value : True if condition else False

# 예시 1) 숫자를 입력받아 10을 더한 값 리턴
f1 = lambda x : x + 10
f1(100)

# 예시 2) 문자열을 입력받아 대문자로 치환
f2 = lambda x : x.upper()
f2('abcd')

# 예시 3) 두 수를 입력받아 두 수의 합 리턴
f3 = lambda x, y : x + y
f3(3, 4)
f3??                                     # 코드 확인

# 예시 4) 두 수를 입력받아 두 수의 합 리턴(인수의 기본값 선언)
#         x가 기본값을 가질 경우 y도 반드시 기본값을 가져야 함
f4 = lambda x = 0, y : x + y             # y 디폴트 생략 불가
f4 = lambda x = 0, y = 0 : x + y         # 정상 작동

# 예시 5) 주민번호를 이용한 성별 출력
jumin = ['8812111223928','8905042323343','90050612343432']
f4 = lambda x : '남' if x[6] == '1' else '여'
list(map(f4, jumin))


# 2) 기본형
#    보다 복잡한 프로그래밍 처리 가능
#    return 객체 정의 필요
def function_name(input_value) :
    body
    return return_value

# 예시 1) 숫자를 입력받아 10을 더한 값 리턴
def f1(x) : 
    return x + 10
f1(100)



# [ 연습 문제 ]
# student.csv 파일을 읽고 과체중 여부 출력
std = pd.read_csv('data/student.csv', encoding = 'cp949')

# 1) map
vweight = []
def f_weight(x, y) :
    if x > (y - 100) * 0.9 :
        vweight = '과체중'
    elif x < (y - 100) * 0.9 :
        vweight = '저체중'
    else :
        vweight = '표준체중'
    return vweight

list(map(f_weight, std['WEIGHT'], std['HEIGHT']))


# 2) for
vtotal = []
for i, j in zip(std['HEIGHT'], std['WEIGHT']) :
    if j > (i - 100) * 0.9 :
        vtotal.append('과체중')
    elif j < (i - 100) * 0.9 :
        vtotal.append('저체중')
    else :
        vtotal.append('표준체중')

std['BMI'] = vtotal                     #컬럼 추가




# =============================================================================
# 적용 함수
# =============================================================================
# 1. 1차원 원소별 적용(in R : sapply, mapply)
# 1) map 함수
#    - 적용 객체 제한 X (리스트, 1차원 배열, 시리즈)
#    - 함수 먼저 나열 > 다중 fetch 가능
l1 = ['1,200', '1,300', '1,400']

f1 = lambda x : int(x.replace(',', ''))
list(map(f1, l1))                        # 결과 확인을 위한 list 선언


# 2) map 메서드
#    - 적용 객체 제한(시리즈만 가능)
#    - 객체 먼저 나열 > 다중 fetch 불가
stud = pd.read_csv('data/student.csv', encoding = 'cp949')

stud['NAME'].map(lambda x : x.startswith('김'))


# 2. 2차원 행별 / 열별 적용(in R : apply)
#    - apply 메서드: 데이터프레임만 가능
#    - axis로 방향 결정(0: 행별/세로 방향, 1: 열별/가로 방향)
a1 = np.arange(1, 17).reshape(4, 4)

a1.sum(axis = 0)
DataFrame(a1).apply(sum, axis = 0)


# 3. 2차원 원소별 적용(in R : apply - margin : c(1,2))
#    - applymap 메서드: 데이터프레임만 가능
#    - if statement, string methods > scalar에만 적용 가능한 경우
df1 = DataFrame(np.array(['a;b', 'c;d', 'e;f', 'g;h']).reshape(2, 2))
f1 = lambda x : x.split(';')[1]
df1.applymap(f1)



# [ 연습 문제 ]
l1 = [1, 2, 3, 4, 5]
l2 = [10, 20, 30, 40, 50]
l3 = ['abc@gmail.com', 'a1234@daum.net']
l4 = ['Smith', 'Allen']
l5 = ['02)345-9984', '031)2245-5683']

# 1) l1에 모든 원소에 10을 더한 결과 리턴
f1 = lambda x : x + 10
list(map(f1, l1))

# 2) l1과 l2 각 원소의 합 리턴
f2 = lambda x, y : x + y
list(map(f2, l1, l2))

# 3) l3에서 각 원소의 이메일 아이디 리턴
f3 = lambda x : x.split('@')[0]
list(map(f3, l3))

# 4) l1**l2
f4 = lambda x, y : x ** y
list(map(f4, l1, l2))

# 5) l5 국번 추출
f5 = lambda x : x.split(')')[1].split('-')[0]
list(map(f5, l5))

# 6) l4와 l5를 이용하여 아래와 같은 형식 리턴
#    이름: Smith, 전화번호: 02)345-9984
f6 = lambda x, y : print('이름: %s, 전화번호: %s' % (x, y))
list(map(f6, l4, l5))




# =============================================================================
# 반복문
# =============================================================================
# 1. for문
for i in value :
    body
    
for i in range(1, 11) :
    print('%2d' % i)
    
    
# 예시 1) 아래 리스트에 대해 각 원소에 10 더한 결과 리턴
l1 = [1, 2, 3, 4, 5]

vresult = []
for i in l1 :
    vresult.append(i + 10)
vresult

# 예시 2) 1 ~ 100 총 합 리턴
vsum = 0
for i in range(1, 101):
    vsum = vsum + i
vsum


# 2. while문
# 예시 1)
i = 1
vsum = 0
while i <= 100 :
    vsum = vsum + i
    i = i + 1
vsum




# =============================================================================
# 중첩 for문
# =============================================================================
for i in range(1, 11) :
    for j in ['a', 'b', 'c'] :
        print('i : %s, j : %s' % (i, j))


# 예시) 아래의 형태로 출력
# 1 2 3
# 4 5 6
# 7 8 9

l1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

for i in l1 :
    for j in i :
        print(j, end = ' ')
    print()
    


# [ 연습 문제 ]
# 아래 중첩 리스트에 대해 2차원 형태로 출력(위치 기반)
l2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] 

# 1.
for i in range(0, 4) :
    for j in range(0, 3) :
        print('%2d' % l2[i][j], end = ' ')
    print()
    
# 2.    
l3 = [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12]] 
for i in range(0, len(l3)) :
    for j in range(0, len(l3[i])) :
        print('%2d' % l3[i][j], end = ' ')
    print()




# =============================================================================
# 조건문
# =============================================================================
if condition :
    True
else :
    False
    
# 예시) v1 값이 10보다 클 경우 5를 더하고, 10보다 작거나 같을 경우 20을 더하여 리턴
v1 = 10

if v1 > 10 :
    v2 = v1 + 5 
else :
    v2 = v1 + 20


# 예시) l1 각 원소가 10보다 클 경우 5를 더하고, 10보다 작거나 같을 경우 20을 더하여 리턴
l1 = [1, 11, 20, 5]

# - 리스트 조건 불가
if l1 > 10 :
    v2 = v1 + 5
else :
    v2 = v1 + 20
    

# - if문 여러 값 동시 전달 불가
from pandas import Series

if Series(l1) > 10 :
    v2 = v1 + 5
else :
    v2 = v1 + 20 
    
# for + if
vresult = []
for i in l1 :
    if i > 10 :
        vresult.append(i + 5)
    else :
        vresult.append(i + 20)
vresult  
        
        
        
#[ 연습 문제 ]
# 부서번호가 10번일 경우 10%, 20이면 11%, 30이면 12% 인상한 결과 리턴
vdeptno = [10, 10, 20, 30]
vsal = [800, 2000, 3000, 2500]

new_sal = []
for i in range(0, len(vsal)) :
    if vdeptno[i] == 10 :
        new_sal.append(round(vsal[i] * 1.1)),
    elif vdeptno[i] == 20 :
        new_sal.append(round(vsal[i] * 1.11)),
    elif vdeptno[i] ==  30 :
        new_sal.append(round(vsal[i] * 1.12))
        
        


# =============================================================================
# np.where 함수
# =============================================================================
# - R의 ifelse 함수와 유사
# - np.where 중복 사용으로 다중 분기 가능

import numpy as np
np.where(condition, True, False)

l1 = [1, 3, 10, 8, 6]
np.where(Series(l1) > 5, 'A', 'B')



# [ 연습 문제 ]
# emp.csv 파일을 읽고 부서별 연봉 증가율에 따른 새 급여 출력
# 10번 부서: 10%, 20번 부서: 11%, 그 외 12%
emp = pd.read_csv('data/emp.csv')
emp.dtypes

# 1) for문
new_sal = []
for i, j in zip(emp['DEPTNO'], emp['SAL']) :
    if i == 10 :
        new_sal.append(round(j * 1.1))
    elif i == 20 :
        new_sal.append(round(j * 1.11))
    else :
        new_sal.append(round(j * 1.12))


# 2) map
def new_sal(x, y) :
    if x == 10 :
        new_sal = round(y * 1.1)
    elif x == 20 :
        new_sal = round(y * 1.11)
    else :
        new_sal = round(y * 1.12)
    return new_sal

list(map(new_sal, emp['DEPTNO'], emp['SAL']))


# 3) np.where
np.where(emp['DEPTNO'] == 10, round(emp['SAL'] * 1.1),
         np.where(emp['DEPTNO'] == 20, round(emp['SAL'] * 1.11),
                  round(emp['SAL'] * 1.12)))

