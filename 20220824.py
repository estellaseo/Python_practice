# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# group by
# =============================================================================
# 분리 > 적용 > 결합
# pandas groupby method
# groupby method 분리만 수행(연산은 따로 적용해야 함)

emp.groupby(by,                          # 분리 컬럼(group by절에 사용될 컬럼)
            axis = 0,                    # 분리 방향
            level,                       # multi index depth
            as_index = True,             # groupby key index 배치 여부
            sort = True,                 # groupby key에 대한 정렬 여부
            group_key)                   # 


# 예시) emp 부서별 급여 평균
emp = pd.read_csv('data/emp.csv')

emp.groupby(by = 'DEPTNO')               # 각 부서별로 분리만 수행
emp.groupby(by = 'DEPTNO').mean()        # 연산대상 선택 X > 모든 숫자 컬럼에 적용됨
emp.groupby(by = 'DEPTNO')['SAL'].mean() # 연산대상 선택 가능

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
emp.set_index(['DEPTNO','EMPNO'])                          # 자동 정렬 X
emp.sort_values(by = 
                'DEPTNO').set_index(['DEPTNO', 'EMPNO'])   # 정렬 후 multi index

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




