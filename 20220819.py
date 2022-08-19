# -*- coding: utf-8 -*-

# =============================================================================
# List Comprehension 리스트 내포 표현식
# =============================================================================
# - 축약형
# - 리스트의 벡터 연산 불가 > for문 적옹 > 객체화 필요(for문 + 객체화의 축약)
[return_value for i in input_value]
[return_value for i in input_value if condition]
[return_value if condition else false for i in input_value]

l1 = [1, 2, 3, 4, 5]

# for문 기본형
l2 = []
for i in l1 :
    l2.append(i + 10)
    
# for문 축약형    
[i + 10 for i in l1]


# 예시 1) l1에서 3보다 큰 경우 10을 더한 결과 리턴 
[i + 10 for i in l1 if i > 3]        # False일 때 생략

# 예시 2) l1에서 3보다 큰 경우 10을 더하고, 아닐 경우 그대로 리턴 
[i + 10 if i > 3 else i for i in l1]



# [ 연습 문제 ]
vsal = ['9,000', '8,000', '8,800']
addr =['aa;bb;cc', 'dd;ee;ff', 'gg;hh;ii']
comm = [100, 200, 300]

# 1) vsal의 10% 인상된 값 출력
[round(int(i.replace(',', '')) * 1.1) for i in vsal]

# 2) addr에서 ;로 분리 구분된 두번째 문자열 추출
[i.split(';')[1] for i in addr]

# 3) comm이 200보다 큰 경우 'A', 작거나 같은 경우 'B' 리턴
['A' if i > 200 else 'B' for i in comm]




# =============================================================================
# Deep Copy, Shallow Copy 깊은 복사, 얕은 복사
# =============================================================================
# - default: 얕은 복사 > 서로 다른 객체명을 가지지만 같은 메모리 영역 공유
# - copy method, 자료구조가 바뀌는 경우 > 깊은 복사 수행

l1 = [1, 2, 3, 4, 5]
l2 = l1                              # 얕은 복사(default)

l2[0] = 10                           # l2 원소값 변경
l2
l1


# - Deep copy
l3 = l1.copy()                       # deep copy case 1

l3[1] = 20                           # l3 원소값 변경
l3
l1                                   # l1 기존값 유지

l4 = l1[:]                           # deep copy case 2

a1 = np.array(l1)                    # 자료구조 array로 변경

a1[-1] = 50
a1
l1




# =============================================================================
# Array 배열
# =============================================================================
# - 다차원 자료구조(1, 2, .. n차원 생성 가능)
# - R의 matrix, array 포함
# - 같은 데이터 타입만 허용
# - numpy 모듈 지원

# 1. 생성
import numpy as np
a1 = np.array([1, 2, 3])
type(a1)

# 2차원 배열 생성(중첩 리스트)
a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# 2. Methods
dir(a2)                              # array 객체에 사용 가능한 method
a1.dtype                             # dtype('int32') 배열을 구성하는 데이터 타입

a2.shape                             # n X m 형태 리턴
a2.shape[0]                          # 행의 수
a2.shape[1]                          # 컬럼 수

a2.ndim                              # 차원 수

a2.flatten()                         # 평탄화(1차원 변경)

a3 = np.arange(1, 13)
a3.shape                             # (12,) 1차원 배열 
a3 = a3.reshape(4, 3)                # 2차원 배열로 변경

a4 = np.arange(1, 25)
a4.reshape(2, 4, 3)                  # 3차원 배열로 변경


# [ 참고 ] 차원의 순서
# 3차원
# in R      : 행(1) > 열(2) > 층(3)
# in Pyhton : 층(0) > 행(1) > 열(2)

# 2차원 
# in R      : 행(1) > 열(2)
# in Pyhton : 행(0) > 열(1)

# 행별, 열별 총합 
a3.sum(axis = 0)                     # 세로 방향: 서로 다른 행끼리(같은 컬럼 내)
a3.sum(axis = 1)                     # 가로 방향: 서로 다른 열끼리(같은 행 내)

# 층별 총합
a4.sum(axis = 0)                     # 서로 다른 층끼리


# 3. Indexing
a3[0, 0]                             # 위치 색인
a3[0, [0, 2]]                        # 리스트를 이용해서 여러 위치 동시 전달
a3[0, 0:2]                           # slice indexing (차원 축소)
a3[0:1, 0:2]                         # 차원 축소 방지
a3[a3 > 5]                           # boolean indexing
a3[[0, 2], [0, 2]]                   # point indexing > a3[0, 0], a3[2, 2]
a3[np.ix_([0, 2], [0, 2])]           # np.ix_() : 여러 행, 여러 컬럼 동시 선택 가능
                                     #            리스트가 아닌 객체 사용 불가

# 차원 축소
a3[:, 1]                             # 2번째 컬럼 선택(차원 축소)
a3[:, 1:2]                           # 슬라이스 선택 시 차원 축소 방지
a3[:, 1].reshape(4, 1)               # 차원 변경을 통해 2차원 유지

a3[:, 1].reshape(4, -1)              # -1 : 열 개수 자동 설정
a3.reshape(2, -1)                    # 2 X 6 자동 설정


# 3차원 배열 색인
a4 = np.arange(1, 25).reshape(2, 3, 4)

# 예시) 6, 7, 18, 19 선택
a4[:, 1:2, 1:3]                      # 3차원 리턴
a4[:, 1, [1, 2]]                     # 2차원 리턴

# 예시) 2, 4, 10, 12,  14, 16, 22, 24 선택
a4[np.ix_([0, 1], [0, 2], [1, 3])]   # np.ix_() 이용을 위해 리스트로 나열 



#[ 연습 문제 ]
# 10부터 10씩 증가하는 25개 숫자를 사용하여 5 X 5 배열 생성 후
a4 = np.arange(10, 260, 10).reshape(5, 5)

# 1) 두번째 컬럼, 네번째 컬럼 선택
a4[:, [1, 3]]

# 2) 120, 140, 150 선택(2차원 형태)
a4[2:3, [1, 3, 4]]
a4[2, [1, 3, 4]].reshape(3, 1)

# 3) 140, 150, 190, 200 선택
a4[2:4, 3:5] 

# 4) 120, 150, 220, 250 선택
a4[np.ix_([2,4], [1,4])]

# 5) 10, 250 선택
a4[[0, 4], [0, 4]]



# 4. 배열 순서
# C: 행 우선순위 
# F: 컬럼 우선순위

np.arange(1, 17).reshape(4, 4)       # default: 행 우선 순위
np.arange(1, 17).reshape(4, 4, order = 'C')
np.arange(1, 17).reshape(4, 4, order = 'F')



# 5. 연산
a5 = np.arange(1, 13).reshape(4, 3)
a6 = np.arange(10, 130, 10).reshape(4, 3)
a7 = np.arange(10, 210, 10).reshape(4, 5)
a8 = np.array([10, 20, 30])
a9 = np.array([10, 20, 30, 40])

a4 + 10                              # 스칼라 연사 가능
a5 + a6                              # 서로 같은 크기를 갖는 배열끼리 연산 가능
a6 + a7                              # 일반적으로 다른 크기 배열끼리 연산 불가


# [ broadcast 기능 ]
# 큰 배열에 작은 배열 연산 시 행마다, 컬럼마다 반복 연산 가능

# 조건
# 1. 작은 쪽 배열이 큰 쪽 배열과 행 또는 컬럼 크기가 같아야 함 
# 2. 작은 쪽 배열은 반드시 1의 크기를 가져야 함
# 3. 크기가 같은 방향 일치

a6 + a8                              # 연산 가능
a6 + a9                              # 연산 불가 (4 X 3) (1 X 4)
a6 + a9.reshape(4, -1)               # 연산 가능 (4 X 3) (4 X 1)



# 6. 산술연산 methods
a5.sum()                             # 전체 총합
a5.sum(axis = 0)                     # 행별(세로) 총합

a5.mean()                            # 전체 평균

a5.min()                             # 전체 최저
a5.max()                             # 전체 최고

a5.var()                             # 전체 분산
a5.std()                             # 전체 표준편차

a5.cumsum()                          # 전체 누적합
a5.cumsum(axis = 0)                  # 세로 누적합 
a5.cumsum(axis = 1)

a5.cumprod()                         # 누적곱(전체)
a5.cumprod(axis = 0)                 # 세로 누적곱


a5.argmax()                          # 최대값 위치(전체 기준)
a5.argmin()                          # 최소값 위치(전체 기준)
a5.argmax(axis =0)

(a5 > 5).sum()                       # True의 수
(a5 > 5).all()                       # 전체 원소값 True 여부
(a5 > 5).any()                       # 하나 이상 True 여부


# 분산
a11 = np.array([1, 2, 3, 4, 5])

((a11 - a11.mean())**2).sum()/(len(a11))
((a11 - a11.mean())**2).sum()/(len(a11) - 1)

a11.var()                            # np > 편차제곱의 합을 n으로 나눈 값
a11.var(ddof = 1)                    #      (n - 1)로 나눈 값과 동일
Series(a11).var()                    # pd > 편차제곱의 합을 (n-1)으로 나눈 값


# [ 참고 ] Skip NA
s2 = Series([1, 2, 3, NA])
s2.sum()                             # default: skipna = True


# [ 참고 ] sum의 다양한 형태
sum([1, 2, 3])                       # 기본 함수
np.sum([1, 2, 3])                    # numpy 함수
np.array([1, 2, 3]).sum()            # numpy method



# 7. 전치 method
# 1) T                               # 행렬 전치
# 2) swapaxes
#    두 축 번호를 전달받아 서로 전치
# 3) transpose
#    0: 행, 1: 열 지정 순서대로 전달


a2.T

a2.swapaxes(0, 1)                    # 인수 순서 중요 X
a2.swapaxes(1, 0)

a2.transpose(0, 1)                   # 인수 순서 중요
a2.transpose(1, 0)                   # 전치

a5 = np.arange(1, 25).reshape(2, 3, 4)
a5.transpose(2, 1, 0)                # 3차원 층과 열 전치
a5.swapaxes(0, 2)                    # 3차원 층과 열 전치



# 8. 파일 입출력
# 1) 외부 파일 불러오기(array로 저장)
np.loadtxt(
    fname,                           # 파일명
    dtype = <class 'float'>,         # 데이터 타입
    delimiter = None,                # 분리구분 기호
    skiprows = 0,                    # 행 생략 여부
    usecols = None,                  # 사용 컬
    ...)

np.loadtxt('data/file1.txt', delimiter = ',')
np.loadtxt('data/file1.txt', delimiter = ',', dtype = 'int')
np.loadtxt('data/file1.txt', delimiter = ',', dtype = 'int', skiprows = 1)
np.loadtxt('data/file1.txt', delimiter = ',', usecols = [0, 2])


# 2) 외부 파일 내려쓰기
np.savetxt(
    fname,                           # 파일명
    X,                               # 저장할 객체
    fmt='%.18e',                     # 데이터 형식
    delimiter=' ',                   # 분리구분 기호
    ...)

np.savetxt('data/file1_0819.txt', a2, fmt = '%d', delimiter = ',', )




# =============================================================================
# NA 확인 함수
# =============================================================================
# - np.isnan 함수: 숫자형 NA에 대해서만 체크 가능
# - pd.isnull 함수, method: 숫자, 문자형 NA에 대해 모두 체크 가능
# - 일반적으로 NA는 float type

emp = pd.read_csv('data/emp.csv')
np.isnan(emp['COMM']).any()
pd.isnull(emp['COMM']).any()

pro = pd.read_csv('data/professor.csv', encoding = 'cp949')
pro['HPAGE']
np.isnan(pro['HPAGE'])               # error(np.isnan 함수 > 숫자형 NA만 체크)
pd.isnull(pro['HPAGE'])              # 정상 작동(pd.isnull > NA 모두 체크 가능)
pro['HPAGE'].isnull()                # method 제공



# [ 연습 문제 ]
# emp.csv 파일을 읽고 comm의 평균(NA인 경우 100으로 수정)
emp = pd.read_csv('data/emp.csv')

# NA 수정 방법
# 1) np.where 이용
np.where(emp['COMM'].isnull(), 100, emp['COMM'])
# 2) 조건 색인 이용
emp.loc[emp['COMM'].isnull(), 'COMM'] = 100

emp['COMM'].mean()




# =============================================================================
# 형 변환 함수
# =============================================================================
# 1. 기본 함수형(20220816.py 참고)
#    - int, float, str
#    - 벡터 연산 불가(scalar만 가능)

# 2. numpy, pandas 제공: astype method
#    - astype method 내부에서 변경할 데이터 타입 전달
#    - 벡터 연산 가능


# 예시) emp에서 EMPNO 문자형 변경
emp = pd.read_csv('data/emp.csv')
emp['EMPNO']                         # Series
emp['EMPNO'].astype('str')

# 예시) emp에서 MGR 정수로 변경
emp['MGR'].astype('int')             # NA(float type)로 인해서 정수 전환 불가



