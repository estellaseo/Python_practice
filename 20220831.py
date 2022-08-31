# -*- coding: utf-8 -*-

# =============================================================================
# stack / unstack, pivot_table
# =============================================================================

# [ 데이터의 형태 ]
# 1) long data 
#    조인 연산
#    groupby 연산(Python은 axis = 1로 설정 가능)

# 2) wide data
#    시각화
#    요약 정보 표현(cross table)
#    행별, 열별 group 연산



# 1. stack
#    wide > long
DataFrame.stack(level = ,          # multi-column의 특정 level를 stack 처리
                dropna = True)     # NA 생략 여부

# 예시 1) 모든 컬럼 stack 처리
df1 = DataFrame(np.arange(1, 17).reshape(4, 4), 
                index = list('abcd'), columns = list('ABCD'))
df1.stack()                        # 각 컬럼 : index의 하위 level로 전달됨
                                   # R : stack 처리 시 df 리턴(variable, values)

# 예시 2) 일부 컬럼 고정, 나머지 stack 처리
#        고정 컬럼 index 설정 후 stack 처리
melt1 = pd.read_csv('data/melt_ex.csv')
melt1.stack()
# multi-index > stack > reset_index
melt1 = melt1.set_index(['year', 'mon']).stack().reset_index()
melt1.rename({'level_2':'name', 0:'qty'}, axis = 1)


# 예시 3) multi-columns DataFrame
#        > level을 이용하여 지정 level stack 처리
df1.columns = [[2007, 2007, 2008, 2008], ['A', 'B', 'A', 'B']]
df1.stack()                        # 하위 level이 default로 stack 처리
df1.stack(level = 0)



# 예시 4) stack 시 NA 처리
df1 = DataFrame(np.arange(1, 17).reshape(4, 4), 
                index = list('abcd'), columns = list('ABCD'))
df1.iloc[0, 1] = NA
df1.stack()                        # default : NA 생략
df1.stack(dropno=a)                # NA 생략 X



# 2. unstack
#    long > wide
DataFrame.unstack(level,           # multi-index에서 상위 level unstack 가능
                  fill_value)

df2 = df1.stack()
df2.unstack()                      # default: 하위 level unstack
df2.unstack(fill_value = 0)        # NA 대체값


# [ 연습 문제 ]
#emp.csv 파일에서 부서별, job별 평균 연봉 교차 테이블 출력
emp = pd.read_csv('data/emp.csv')
emp.groupby(['DEPTNO', 'JOB'])['SAL'].mean().unstack()



# 3. pivot / pivot_table
#    unstack(cross table 생성) 처리 용이
df.pivot_table(values = ,          # value 컬럼 대상
               index = ,           # index 방향에 배치할 대상
               columns = ,         # column 방향에 배치할 대상
               aggfunc = 'mean',   # 결합함수(aggregate function)
               fill_value = ,      # NA 대체값
               margins = False,    # 총계 출력 여부
               dropna = False,     # NA 출력 여부
               ...)

emp.pivot_table(index='DEPTNO', columns='JOB', values='SAL', margins=True)

#[ 참고 ] pivot과 pivot_table의 차이
emp.pivot(index, columns, values)  # 제한된 기능을 갖는 pivot 함수(aggfunc 불가)


# 예시) 리스트로 복수 대상 전달
kimchi = pd.read_csv('data/kimchi_test.csv', encoding = 'cp949')
kimchi.pivot_table(index = '판매년도', columns = '제품', 
                   values = ['수량', '판매금액'], aggfunc = 'sum')


# [ 연습 문제 ]
# movie_ex1.csv 파일을 읽고
movie = pd.read_csv('data/movie_ex1.csv', encoding = 'cp949')

# 1) 지역-시도별, 성별 이용비율의 총합을 정리한 교차 테이블 생성
movie.pivot_table(index = '지역-시도', columns = '성별', 
                  values = '이용_비율(%)', aggfunc = 'sum')

# 2) 일별, 연령대별 이용비율의 총합을 정리한 교차테이블 생성
movie.pivot_table(index = '일', columns = '연령대', 
                  values = '이용_비율(%)', aggfunc = 'sum')

# 3) 연령대별 이용비율이 가장 높은 성별 확인
movie_sum = movie.groupby(['연령대', '성별'], axis = 0)['이용_비율(%)'].sum()
movie_sum[movie_sum.groupby('연령대').idxmax()]

movie_sum2 = movie.pivot_table(index = '연령대', columns = '성별', 
                               values = '이용_비율(%)', aggfunc = 'sum')
movie_sum2.idxmax(axis = 1)






