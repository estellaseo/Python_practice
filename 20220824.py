# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# GROUP BY
# =============================================================================
# 분리 > 적용 > 결합
# pandas groupby method
# groupby method 분리만 수행(연산은 따로 적용해야 함)

emp.groupby(by = ,                       # 분리 컬럼(group by절에 사용될 컬럼)
            axis = 0,                    # 분리 방향
            level = ,                    # multi index depth
            as_index = True,             # groupby key index 배치 여부
            sort = True,                 # groupby key에 대한 정렬 여부
            group_keys = True)           # groupby key 출력 여부


# 예시) emp 부서별 급여 평균
emp = pd.read_csv('data/emp.csv')

emp.groupby(by = 'DEPTNO')               # 각 부서별로 분리만 수행
emp.groupby(by = 'DEPTNO').mean()        # 연산대상 선택 X > 모든 숫자 컬럼에 적용됨
emp.groupby(by = 'DEPTNO')['SAL'].mean() # 연산대상 선택 가능
# [ 참고 ]
emp['SAL'].groupby(emp['DEPTNO']).mean() # 연산대상만 선택해서 groupby 수행 가능

# 예시 ) emp 부서별 급여, COMM 평균
# 다중 컬럼 선택을 통해 여러 연산 가능
emp.groupby(by = 'DEPTNO')[['SAL', 'COMM']].mean()

# 예시 ) emp 부서별, 직무별 급여 평균
# multi-index를 갖는 Series로 리턴
emp.groupby(by = ['DEPTNO', 'JOB'])['SAL'].mean()

# reset_index() OR as_index = False
emp.groupby(by = ['DEPTNO', 'JOB'])['SAL'].mean().reset_index()
emp.groupby(by = ['DEPTNO', 'JOB'], as_index = False)['SAL'].mean()


# [ 참고 ]  multi index
emp.set_index(['DEPTNO','EMPNO'])                        # 자동 정렬 X
emp.sort_values(by = 
                'DEPTNO').set_index(['DEPTNO', 'EMPNO']) # 정렬 후 multi index

# python 색인 함수 및 메서드
# - np.ix_     : array 색인 함수
# - iloc, loc  : pandas 색인 메서드
# - xs         : pandas 색인 메서드(multi index 접근)



# [ 연습 문제 ]
# 1. 학년별 평균 시험 성적 출력
std = pd.read_csv('data/student.csv', encoding = 'cp949')
exam = pd.read_csv('data/exam_01.csv', encoding = 'cp949')

std.merge(exam).groupby(by = 'GRADE')['TOTAL'].mean()


# [ 참고 ] groupby 분리 과정 > groupby key, data 각각 분리
for i, j in std.merge(exam).groupby(by = 'GRADE') :
    print('key : %s' % i)
    print(j)


# 2. 학년별 평균 시험 성적보다 높은 성적을 받는 학생의 이름, 학년, 성적 출력
std = std.merge(exam)
std2 = std.merge(exam).groupby(by = 'GRADE', as_index = False)['TOTAL'].mean()
# sol 1)
vtotal = std['GRADE'].map(lambda x : 
                         std2.loc[x == std2['GRADE'], 'TOTAL'].iloc[0])
std.loc[std['TOTAL'] > vtotal, ['NAME', 'GRADE', 'TOTAL']]

# sol 2)
std3 = pd.merge(std, std2, on = 'GRADE')
std3.loc[std3['TOTAL_x'] > std3['TOTAL_y'], ['NAME', 'GRADE', 'TOTAL_x']]


# 3. 각 학년별로 시험성적이 가장 낮은 학생의 정보 출력
std_min = std.groupby(by = 'GRADE', as_index = False)['TOTAL'].min()
pd.merge(std, std_min, on = ['GRADE', 'TOTAL'])



# [ groupby 기능 ]
# 1. 가로방향 그룹 연산, 외부객체를 사용한 그룹핑
#    - axis = 1
df1 = DataFrame(np.arange(1, 25).reshape(4, 6), columns = list('abcdef'))
g1 = ['a1', 'a1', 'b1', 'b1']
g2 = ['a1', 'a1', 'a1', 'b1', 'b1', 'b1']

df1.groupby(g1).sum()
df1.groupby(g2, axis = 1).sum()          # 가로방향 그룹 연산


# 2. 서로 다른 함수의 전달
#    - agg method 활용

# 예시) 부서별 급여의 총 합, COMM의 평균
# 하나의 연산 함수만 전달 가능
emp.groupby('DEPTNO')[['SAL', 'COMM']].mean()
# agg method를 이용하여 각 연산 컬럼에 대해 두 함수 모두 연산
emp.groupby('DEPTNO')[['SAL', 'COMM']].agg(['sum', 'mean'])
# 각 연산 컬럼별 서로 다른 함수 전달 : dict 활용
d1 = {'SAL' : 'sum', 'COMM' : 'mean'}
emp.groupby('DEPTNO')[['SAL', 'COMM']].agg(d1)


# 3. 사용자 정의 함수 전달
#    - groupby 객체를 위한 apply method 활용

# groupby + sort_values
# 예시) 부서별 SAL 높은순 정렬
emp.groupby('DEPTNO').sort_values('SAL') # groupby 객체에 직접 sort_values 불가
type(emp.groupby('DEPTNO'))              # DataFrameGroupBy

# 1) 정렬: 1차 부서번호, 2차 급여
emp.sort_values(['DEPTNO', 'SAL'], ascending = [True, False])
# 2) apply method: 각 부서별 데이터 fetch
emp.groupby('DEPTNO').apply(lambda x : x.sort_values('SAL', ascending = False))
# group key 생략 - 구분 가능한 group key를 컬럼에 이미 포함하고 있는 경우에만
emp.groupby('DEPTNO', 
            group_keys = False).apply(lambda x : 
                                      x.sort_values('SAL', ascending = False))




# [ 참고 ] 차원 축소
# 2차원 객체에서 하나의 행, 컬럼 선택 > 1차원
# 리스트, 시리즈에서 하나의 원소 > scalar

# 모델링 시 변수가 많을수록 복잡한 모델 생성 > 과대적합 가능성 높아짐 > 모델 단순화 필요
# 이 때 차원축소가 필요함

emp.iloc[0, :]                           # 차원축소 발생(Series 리턴)
emp.iloc[0:1, :]                         # 차원축소 방지(DataFrame 리턴)

emp.loc[:, 'ENAME']                      # 차원축소 발생(Series 리턴)
emp.loc[:, 'ENAME':'ENAME']              # 차원축소 방지(DataFrame 리턴)

emp['ENAME']                             # 차원축소 발생(Series 리턴)
emp[['ENAME']]                           # 차원축소 방지(DataFrame 리턴)

emp.groupby(by = 'DEPTNO')['SAL'].mean() # 차원축소 발생(Series 리턴)
emp.groupby(by = 'DEPTNO')[['SAL']].mean()           #(DataFrame 리턴)



# [ 연습 문제 ]
# delivery.csv 파일을 읽고 시간대별 배달이 가장 많은 음식업종 추출
deli = pd.read_csv('data/delivery.csv', encoding = 'cp949')
# 시간대별 업종별 통화건수 총합
deli_total = deli.groupby(by = ['시간대', '업종'], 
                          as_index = False)['통화건수'].sum()
# 시간대별 최다 통화건수
deli_max = deli_total.groupby(by = '시간대')['통화건수'].max()
# 시간대별 최대 통화건수에 해당하는 업종
pd.merge(deli_total, deli_max, on = ['시간대', '통화건수'])



# [ 연습 문제 ]
# sales.csv 파일을 읽고 
s1 = pd.read_csv('data/sales2.csv', encoding = 'cp949')

# 1) 날짜별 출고, 판매, 반품 총합
s1.groupby(by = '날짜')['출고', '판매', '반품'].sum()

# 2) 지점별, 품목별 출고 평균
s1.groupby(by = ['지점', '품목'])['출고'].mean()

# 3) 품목별 출고 평균, 판매의 총합
d2 = {'출고' : 'mean', '판매' : 'sum'}
s1.groupby(by = '품목')[['출고', '판매']].agg(d2)

# 4) 날짜별, 지점별로 출고가 높은순 정렬
s1.sort_values(['날짜', '지점', '출고'], ascending = [True, True, False])

f1 = lambda x : x.sort_values('출고', ascending = False)
s1.groupby(by = ['날짜', '지점'], group_keys = False).apply(f1)




# =============================================================================
# rank
# =============================================================================
# pandas 제공 함수
# Series, DataFrame 이용 가능

s1 = Series([1, 3, 5, 7, 10 ,2, 6])
s1.sort_values()[:3]

s1.rank(axis = 0,                        # 순서매김 방향  
        method = 'average',              # 동순위 처리 (min, max, first, dense)
        na_option = 'keep',              # NA 처리 방식
        ascending = True)                # 정렬 방식

s1[s1.rank(method = 'min') <= 3]


s2 = Series([1, 3, 10, 2, 5, 7, 10])
s2.rank(ascending = False)               # default: method = 'average'
s2.rank(ascending = False, method = 'min')                 # 공동 1위
s2.rank(ascending = False, method = 'max')                 # 공동 2위
s2.rank(ascending = False, method = 'first')               # 순서대로 1, 2위
s2.rank(ascending = False, method = 'dense')               # 공동 1위 다음 2위


# 데이터프레임에서 순위 부여 방식
a1 = [1, 4, 3, 6, 4, 5, 7, 7, 9]
df1 = DataFrame(np.array(a1).reshape(3, 3))
df1.rank(axis = 0)                       # 세로방향 fetch > 각 컬럼 내 순위 리턴
df1.rank(axis = 1)                       # 가로방향 fetch > 각 행 내 순위리턴




# [ 연습 문제 ]
# 지역-시도별로 영화 이용비율이 가장 높은 2개 연령대 추출
movie = pd.read_csv('data/movie_ex1.csv', encoding = 'cp949')
mov_total = movie.groupby(by = ['지역-시도', '연령대'], 
                          as_index = False)['이용_비율(%)'].sum()
# 1) 정렬
f3 = lambda x : x.sort_values('이용_비율(%)', ascending = False)[:2]
mov_total.groupby(by = '지역-시도', group_keys = False).apply(f3)

# 2) 순위
mov_total['rank'] = mov_total.groupby(by = '지역-시도')['이용_비율(%)'].rank(ascending = False)
mov_total.loc[mov_total['rank'] <= 2, :]



