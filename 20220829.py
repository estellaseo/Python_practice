# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# Python과 DB 연동(Oracle)
# =============================================================================
# RDBMS : Oracle, DB2, ms-sql, my-sql, tibero
# NOSQL : mongoDB, ...


# 1. oracle 접속 모듈 설치(cx_Oracle)
# pip install cx_Oracle

import cx_Oracle

# [ 참고 ] postgreSQL, mysql(pymysql)



# 2. DB 접속 정보
#    - IP   : localhost
#    - port : 1521
#    - DB name(Service name) : orcl
#    - username : scott
#    - passwd : oracle


# connection info : 'username/passwd@IP:port/dbname'
con1 = cx_Oracle.connect('scott/oracle@localhost:1521/orcl')
con2 = cx_Oracle.connect('scott/oracle@192.168.17.31:1521/orcl')



# 3. sql 실행
vsql = 'select * from emp where deptno = 10'
vsql1 = 'select * from student where grade = 1'
vsql2 = 'select * from t_hong'

emp = pd.read_sql(vsql, con = con1)
std = pd.read_sql(vsql1, con = con1)
test = pd.read_sql(vsql2, con = con2)



# =============================================================================
# shift
# =============================================================================
# - pandas 제공(Series, DataFrame에 사용)
# - 행과 열의 이동

s1 = Series([1, 6, 5 , 2, 10])
s1.shift(periods = 1,             # 가져올 값의 위치(이전 값 가져오기)
         freq = ,                 # 날짜 빈도(일주일 이전, 한 달 이전, ...)
         axis = 0                 # 방향
         fill_value = )           # NA 대신 리턴 값

s1.shift()
s1.shift(-1)                      # 이후값 가져오기
s1.shift(2)                       # 이전 * 2 값 가졍
s1.shift(2, fill_valu = 0)


df1 = DataFrame(np.arange(1, 10).reshape(3, 3))
df1.shift()
df1.shift(axis = 1)               # 이전 컬럼값 가져오기



# [ 연습 문제 ]
# card_history.csv 파일을 읽고
card = pd.read_csv('data/card_history.csv', encoding = 'cp949')
card = card.set_index('NUM')
card = card.applymap(lambda x : int(x.replace(',', '')))

# 1. 식료품에 대해 이전일자 대비 지출증가율 출력
(card['식료품'] - card['식료품'].shift()) / card['식료품'].shift() * 100

# 2. 각 지출품목에 대해 이전일자 대비 지출증가율 출력
((card - card.shift()) / card.shift() * 100).fillna(0)




# =============================================================================
# 중복값 처리
# =============================================================================
s1 = Series([1, 3, 1, 4, 5, 5])
s1.unique()                       # array return
s1.duplicated()                   # 중복 여부 확인 가능
s1[s1.duplicated()]               # 중복값 확인
s1[~s1.duplicated()]              # 중복값 제거한 unique value(Series)

s1.drop_duplicates(subset,        # 특정 그룹(부분집합)이 중복될 경우 ㅈ)
                   keep,
                   inplace, 
                   ignore_index)


df2 = DataFrame({'A' : [1, 2, 3, 4], 'B' : [2, 2, 5, 6], 'C' : [3, 3, 7, 8]})
df3 = DataFrame({'A' : [1, 2, 3, 4], 'B' : [2, 2, 5, 6], 'C' : [3, 3, 7, 8]})

df3.drop_duplicates()             # 모든 컬럼의 값이 같은 행을 제거(첫버ㄴ=)
df3.drop_duplicates(keep = 'last')# 두번째 행 남음
df3.drop_duplicates(ignore_index = True)  # index 초기화





# =============================================================================
# WRAP-UP
# =============================================================================
# 1. 기본 자료구조 : list, dictionary, array, series, dataframe, (tuple, set)

# 2. 색인 : numpy(), pandas(iloc, loc, xs)
a1[np.ix_([1, 4], [4, 6])]

# 3. 정렬 : sort_index, sort_values

# 4. 그룹연산 : groupby(apply, agg)

# 5. 조인 : pd.merge

# 6. 결측치 처리
#    1) NA 확인(np.isnan, pd.isnull, s1.isnull)
pd.isnull('a') 
#    2) NA로 치환(replace)
df1.replace(['.', '!', '?'], NA)
#    3) NA를 치환(fillna)
#    4) NA 삭제(dropna)

# 7. 이상치 처리 :
df1.loc[df1['a'] >= 100, 'a'] = 100

# 8. 중복값 처리(unique, duplicated, drp_duplicates)

# 9. 시계열 처리(날짜포맷변경(strftime), 날짜파싱(strptime))

# 10. 유용한 기타 method : shift(), ...

# 응용문법
# 1. multi-index
# 2. stack / unstack
# 3. regex
# 4. sqldf

