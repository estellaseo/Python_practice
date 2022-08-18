# -*- coding: utf-8 -*-

# 1. 구구단 출력
for i in range(1,10) :
    for j in range(2,10) :
        print('%d X %d = %2d' % (j, i, i*j), end = '  ')
    print()



# 2. 별 출력
for i in range(1, 10) :
    if i < 6 :
        print('  ' * (5 - i), '\u2605' * (i * 2 - 1))
    else :
        print('  ' * (i - 5), '\u2605' * (-i * 2 + 19 ))



# 3. 다음의 리스트 생성 후 연산
ename = ['smith','allen','king'] 
jumin = ['8812111223928','8905042323343','90050612343432']
tel=['02)345-4958','031)334-0948','055)394-9050','063)473-3853']
vid=['2007(1)','2007(2)','2007(3)','2007(4)']

# 1) ename에서 i를 포함하는지 여부 확인
f1 = lambda x : 'i' in x
list(map(f1, ename))

# [ 참고 ] boolean indexing
ename[[True, False, True]]             #리스트 객체를 이용한 색인 불가

from pandas import Series
Series(ename)[list(map(f1, ename))]    #시리즈 객체로 변환 후 색인


# 2) jumin을 사용한 성별 출력
for i in jumin :
    if i[6] == '1' :
        print(i[6], '남')
    else :
        print(i[6], '여')


# 3) ename에서 smith 또는 allen 인지 여부 출력 [True,True,False]
for i in ename :
    if i == 'smith' :
        print(True)
    elif i == 'allen' :
        print(True)
    else :
        print(False)
        
#[ 참고 ] isin 메서드(Series, Dataframe에만 적용 가능)
Series(ename).isin(['smith', 'allen'])


# 4) tel에서 다음과 같이 국번 XXX 치환 (02)345-4958 => 02)XXX-4958)
f2 = lambda x : x[0:x.find(')') + 1] + \
    'X' * len(x.split(')')[1].split('-')[0]) + \
        x[x.find('-'):]
list(map(f2, tel))


# 5) vid 에서 각각 년도(vyear)와 분기(vqt)를 따로 저장
f3 = lambda x : x.split('(')[0]
vyear = list(map(f3, vid))

f4 = lambda x : x.split('(')[1][0]
vqt = list(map(f4, vid))



# 4. 다음의 리스트 생성 후 연산  
vsal = ['1,100','1,200','2,200','3,000','5,000']
vcomm = [ 300, 200, 100, 500, 800]

# 1) 10%  인상된 연봉을 구하세요
f5 = lambda x : round(int(x.replace(',', '')) * 1.1)
sal = list(map(f5, vsal))


# 2) sal + comm값을 구하세요
# sol 1.
f6 = lambda x, y : x + y
list(map(f6, sal, vcomm))

# sol 2. 위치 기반 for문
vtotal = []
for i in range(0, len(sal)) :
    vtotal.append(sal[i] + vcomm[i])

# sol 3. zip 함수 이용

# =============================================================================
# zip 함수
# =============================================================================
# 여러 객체를 하나씩 fetch

list(zip([1, 2, 3, 4, 5], ['a', 'b', 'c']))
list(zip([1, 2, 3, 4, 5], 'abcde'))

vtotal = []
for i, j in zip(sal, vcomm) :
    vtotal.append(i + j)


