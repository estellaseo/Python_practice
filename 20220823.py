# -*- coding: utf-8 -*-
run my_profile
from pandas import DataFrame

# =============================================================================
# rename 메서드
# =============================================================================
# - index object(index, columns) 일부 수정을 위한 메서드
# - pandas 제공
# - axis로 방향 전달(0: index(default), 1: columns)

df.rename(mapper,                         # mapping object(dict {old : new})
           axis = 0)                      # 수정 방향

# 예시)
df2.rename({'2013년' : '2022년'}, axis = 0)


 

# =============================================================================
# fillna 메서드
# =============================================================================
# NA 치환 메서드
# pandas 제공(Series, DataFrame만 적용 가능)
# 스칼라, 시리즈, 딕셔너리 등을 전달하여 치환 가능
# 이전값, 이후값(위치 기준) 치환 가능

df1 = DataFrame(np.arange(1, 26).reshape(5, 5), columns = list('abcde'))
df1.iloc[0, 0] = NA
df1.iloc[1, 1] = NA
df1.iloc[2, 2] = NA
df1.iloc[3, 3] = NA

# 1) 모두 같은 값으로 치환
df1.fillna(0).astype('int')


# 2) 컬럼마다 서로 다른 값으로 치환 
df1['a'].fillna(10)
df1['b'].fillna(df1['b'].mean())

df1.fillna({'a' : 0, 'b' : 1, 'c' : 2})   # 각 컬럼별 수정값 다르게 전달 가능


# 3) 이전, 이후값 치환 가능
df1.fillna(method = 'ffill')              # 이전 행의 값으로 치환(axis = 0)
df1.fillna(method = 'ffill', axis = 1)    # 이전 컬럼 값으로 치환

df1.fillna(method = 'bfill')              # 이후 값으로 치환


# 4) 시리즈를 사용한 치환(수정 시 다른 대상(시리즈, 데이터프레임) 참조)
df1['d'].fillna(df1['b'])

df2 = DataFrame(np.arange(10, 251, 10).reshape(5, 5), columns = list('abcde'))
df1.fillna(df2)                           # 같은 위치의 값으로 수정



# [ 연습 문제 ]
# student.csv 파일을 읽고, DEPTNO2가 없을 때 DEPTNO1로 표현
stud = pd.read_csv('data/student.csv', encoding = 'cp949')
stud['DEPTNO2'].fillna(stud['DEPTNO1']).astype('int')



# [ 연습 문제 ]
# read_test.csv 파일을 읽고 
t1 = pd.read_csv('data/read_test.csv')

# 1) a 컬럼의 평균 출력(단, 결측치가 있을 경우 최솟값으로 수정)
t1['a'].mean()                            # type error(str)
t1['a'].astype('int')                     # '.' 때문에 숫자 변경 불가
t1['a'].replace('.', NA).astype('float')  # '-' 때문에 숫자 변경 불가

t1['a'].replace(['.', '-'], NA)           # 여러 값을 동시에 치환 가능
t1['a'].replace('[.!?-]', NA, regex = True)
t1['a'].replace('\W', NA, regex = True)   # '[\W]' : 숫자와 문자가 아닌

# . 또는 - 결측치 처리
t1['a'] = t1['a'].replace(['.', '-'], NA).astype('float')

# 결측치 대치
t1['a'] = t1['a'].fillna(t1['a'].min())
t1['a'].isnull().sum()

# 평균 출력
t1['a'].mean(skipna = False)              # skipna = False로 NA 여부 마지막 확인


# 2) b 컬럼의 평균 출력(단, 190 초과값은 이상치이며 이상치의 경우 190으로 수정,
#                     결측치는 최솟값으로 수정)
t1['b'].astype('int')
# 결측치 처리
t1['b'] = t1['b'].replace(['?', '!'], NA).astype('float')
# 이상치 처리
(t1['b'] > 190).sum()                     # 이상치 개수 확인
t1.loc[t1['b'] > 190, 'b'] = 190          # 이상치 대치
# 결측치 대치
t1['b'] = t1['b'].fillna(t1['b'].min())
# 평균
t1['b'].mean()

dir(t1)                                   # dir 객체 전달 시 호출 가능 메서드 목록
dir(pd)                                   # dir 모듈 전달 시 모듈 포함된 함수 목록




# =============================================================================
# 정렬
# =============================================================================
# 시리즈, 데이터프레임 정렬
# pandas 제공 메서드
# sort_index, sort_values

# 1. sort_index : index object 정렬
emp = pd.read_csv('data/emp.csv')
emp = emp.set_index('EMPNO')

emp.sort_index(axis = 0,                  # 행 or 컬럼 선택
               ascending = True)          # 정렬 순서(순방향, 역방향)

emp.sort_index(axis = 0, ascending = False)
emp.sort_index(axis = 1)


# 2. sort_values : 특정 값(or 컬럼값 in DataFrame)으로 정렬 
emp.sor_values(by,                        # 정렬 컬럼(여러개 전달 가능 - 리스트)
               axis = 0,                  # 정렬 방향
               ascending = True,          # 정렬 순서(여러개 전달 가능)
               inplace = False,           # 원본 수정 여부
               np_position = 'last'       # NA 배치 순서
               ignore_index = False)      # index 정렬 초기화

emp.sort_values(by = 'SAL', ascending = False)
emp.sort_values(by = 'SAL', ascending = False, inplace = True)
emp.sort_values(by = 'COMM', na_position = 'first')

emp.sort_values(by = ['DEPTNO', 'SAL'], ascending = [True, False])
               
emp.reset_index()                         # 설정한 index 본문 컬럼으로 복구
emp = emp.reset_index()

emp.sort_values(by = 'DEPTNO', ignore_index = True)



# [ 연습 문제 ]
# student.csv 파일을 읽고 
std = pd.read_csv('data/student.csv', encoding = 'cp949')
std = std.loc[:, ['NAME', 'JUMIN', 'HEIGHT']]

# 1) 성별 컬럼 추가(G1 컬럼: M/F, G2 컬럼: 남자/여자)
m1 = std['JUMIN'].map(lambda x : str(x)[6])
std['G1'] = np.where(m1 == '1', 'M', 'F')
std['G2'] = m1.replace({'1' : '남자', '2' : '여자'})

# 2) 성별 순서대로 정렬, 같은 성별 내에서 키가 높은 순으로 정렬
std.sort_values(by = ['G1', 'HEIGHT'], ascending = [False, False])
std.sort_values(by = ['G2', 'HEIGHT'], ascending = [True, False])
# 문자 컬럼과 숫자 컬럼이 동시전달 시 서로 다른 순서 보장됨




# =============================================================================
# merge 결합
# =============================================================================
# pd.merge 함수 제공
# equi join만 가능
# 두 객체에 대한 결합만 가능 
# outer join 지원 

pd.merge(left,
         right,
         how = 'inner',                   # inner or outer join(left, right)
         on = ,                           # join 컬럼(동일한 이름의 컬럼으로 조인)
         left_on = ,                      # 1st 객체 조인 컬럼명
         right_on = ,                     # 2nd 객체 조인 컬럼명
         left_index = False,              # 1st 객체 인덱스 조인 컬럼 사용 여부
         right_index = False,             # 2nd 객체 인덱스 조인 컬럼 사용 여부
         sort = False,                    # 조인키 정렬 여부
         suffixes = ('_x', '_y'))         # 동일 컬럼명 구분 기호



# 예시) emp, dept join
emp = pd.read_csv('data/emp.csv', encoding = 'cp949')
dept = pd.read_csv('data/dept.csv', encoding = 'cp949')

pd.merge(emp, dept)                       # natural join 수행
pd.merge(emp, dept, sort = True)          # join key 정렬
pd.merge(emp, dept, on = 'DEPTNO')        # 조인 컬럼 전달


# 예시) 각 학생의 이름, 학년, 지도교수이름(없는 경우 NA)
std = pd.read_csv('data/student.csv', encoding = 'cp949')
pro = pd.read_csv('data/professor.csv', encoding = 'cp949')

# std 기준 outer join > left outer join
pd.merge(std, pro, how = 'left', on = 'PROFNO')


# left_index, right_index 활용
dept = dept.set_index('DEPTNO')

# 예시) emp에서 EMPNO 컬럼, dept는 index 값으로 조인
pd.merge(emp, dept, left_on = 'DEPTNO', right_index = True)



# [ 연습 문제 ]
# emp.csv 파일을 읽고 각 직원의 이름, 급여, 부서번호, 상위관리자 이름, 급여 출력
# 상위관리자가 없는 경우 본인 이름과 본인 급여 출력
emp = pd.read_csv('data/emp.csv')
emp.columns
# 필요한 컬럼으로 새 데이터프레임 만들기
emp1 = emp.loc[:, ['EMPNO', 'ENAME', 'SAL', 'DEPTNO', 'MGR']]
emp2 = emp.loc[:, ['EMPNO', 'ENAME', 'SAL']]
# 결합 후 출력
emp3 = pd.merge(emp1, emp2, left_on = 'MGR', right_on = 'EMPNO', how = 'left')
emp3 = emp3.loc[:, ['ENAME_x', 'SAL_x', 'DEPTNO', 'ENAME_y', 'SAL_y']]
# 결측치 처리 case 1 - 컬럼 각각 설정하기
emp3['ENAME_y'] = emp3['ENAME_y'].fillna(emp3['ENAME_x'])
emp3['SAL_y'] = emp3['SAL_y'].fillna(emp3['SAL_x'])
# 결측치 처리 case 2 - 컬럼 재배치 후 이전 컬럼 값 가져오기
emp3 = emp3[['ENAME_x', 'ENAME_y', 'SAL_x', 'SAL_y', 
             'DEPTNO']].fillna(method = 'ffill', axis = 1)

emp3.columns = ['ENAME', 'SAL', 'DEPTNO', 'MGR_ENAME', 'MGR_SAL']



# [ 연습 문제 ] non equi join
# gogak, gift를 사용하여 각 고객이 가져갈 수 있는 상품 목록 출력
g1 = pd.read_csv('data/gogak.csv', encoding = 'cp949')
g2 = pd.read_csv('data/gift.csv', encoding = 'cp949')

f_goods = lambda x : g2.loc[x < g2['G_END'], 'GNAME'].iloc[0]
prelist = g1['POINT'].map(f_goods)




# =============================================================================
# SQL 문법 사용하기
# =============================================================================
# SQL 문법을 사용하여 데이터 선택, 수정, 연산, 조인 등을 수행
# non-equi join을 SQL 문법을 사용하여 처리 가능
# 외부 모듈(패키지)로 사용
# pip install pandasql 설치후 이용

from pandasql import sqldf
sqldf('select * from emp')
sqldf("select * from emp where ename =  'SMITH'")
sqldf('select * from emp e, dept d where e.deptno = d.deptno')  # oracle 지원 X
sqldf('select * from emp e join dept d on e.deptno = d.deptno') # Ansi 표준

# non-equi join 수행
vsql = '''
       select g1.gname '고객이름', g2.gname '상품명'
         from g1 join g2
           on g1.point between g2.g_start and g2.g_end
       '''
sqldf(vsql)




