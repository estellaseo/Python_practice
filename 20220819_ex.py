# -*- coding: utf-8 -*-
import numpy as np

# 1. emp.csv 파일을 array로 불러온 후(컬럼명 제외)
emp = np.loadtxt('data/emp.csv', delimiter = ',', dtype = 'str', skiprows = 1)

# 1) sal이 3000이상인 행 선택
emp[emp[:, 5].astype('int') >= 3000, :]

# 2) comm이 없는 직원의 이름, 부서번호, comm 선택
emp[np.ix_(emp[:, 6] == '', [0, 1, 6])]

# 3) 이름이 S로 시작하는 직원의 이름, 사번, sal 선택
emp[np.ix_(emp[:, 1] == 'SMITH', [1, 0, 5])]
np.where(emp[:, 1].startwith('S'))

# 4) 이름이 smith와 allen의 이름, 부서번호, 연봉 출력
emp[np.ix_((emp[:, 1] == 'SMITH') | (emp[:, 1] == 'ALLEN'), [1, -1, 5])]

# 5) deptno가 30번 직원의 comm의 총 합
emp[:, 6] = np.where(emp[:, 6] == '', 0, emp[:, 6])
emp[emp[:, -1] == '30',6].astype('int').sum()



# 2. 1부터 증가하는 3 X 4 X 5 배열 생성 후
a1 = np.arange(1, 61).reshape(3, 4, 5)

# 1) 모든 값에 짝수는 *2를 홀수는 *3을 연산하여 출력
np.where(a1%2 == 0, a1 * 2, a1 * 3)

# 2) 각 층의 첫번째 세번째 행의 두번째 네번째 컬럼 선택하여 NA로 치환
a1[np.ix_([0, 1, 2], [0, 2], [1, 3])].astype('float')



# 3. 다음의 배열을 생성한 후
a1 = np.array([[1,500,5],[2,200,2],[3,200,7],[4,50,9]])

# 1) 위 배열에서 두 번째 컬럼 값이 300 이상인 행을 선택
a1[a1[:, 1] >= 300, :]

# 2) 세 번째 컬럼 값이 최대인 행을 선택
a1[a1[:, 2] == a1[:, 2].max(), :]



# 4. 1부터 시작하는 3X4의 배열을 생성한 후
a2 = np.arange(1, 13).reshape(3, 4)

# 1) 각 행별 분산 연산(axis=0), 직접 분산을 구하는 수식으로 계산, var 메서드 값과 비교
# var method : array([10.66666667, 10.66666667, 10.66666667, 10.66666667])
a2.var(axis = 0)

# 연산
[((a2[:, i] - a2.mean(axis = 0)[i])**2).sum()/len(a2[:, i]) 
 for i in range(0, (len(a2[:,0]) + 1))]

(a2 - a2.mean(axis = 0)**2).sum()/4


# 2) 각 컬럼별 분산 연산(axis=1), 직접 분산을 구하는 수식으로 계산, var 메서드 값과 비교 
#var method : array([1.25, 1.25, 1.25])
a2.var(axis = 1)

# 연산
[((a2[i, :] - a2.mean(axis = 1)[i])**2).sum()/len(a2[:, i]) 
 for i in range(0, (len(a2[i, :]) + 1))]


((a2[0, :] - a2.mean(axis = 1)[0])**2).sum()/len(a2[:, 0])

