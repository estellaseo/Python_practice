# -*- coding: utf-8 -*-
run my_profile

#[ 참고 ] DataFrame이 호출하는 apply와 groupby 객체가 호출하는 apply 차이
# f1의 x : Series 전달
df1.apply(func, axis, raw, ...)
# f1의 x : DataFrame 전달
df1.groupby('회원번호').apply(func, *args, **kwargs) 



# =============================================================================
# 정규표현식 wrap up
# =============================================================================
# ^      : 시작
# $      : 끝
# [ ]    : 여러 값 동시 전달, 연속 패턴(ex. 숫자 : [0-9], 한글 : [가-힣])
# .      : 공백 포함 한 글자
# +      : 1회 이상
# \      : 특수기호 일반 기호화(ex. \. : . 그대로 전달(한 글자의 의미 X))
# {}     : 추출 그룹 형성
# {i, j} : i회 이상 j회 이하(ex. {3,} : 3회 이상)

# \d     : 숫자
# \s     : 공백 or 탭
# \w     : 문자 또는 숫자
# \W     : 문자 또는 숫자가 아닌 특수기호 등



s1 = Series(['abcd', 'Aadc', '1234aa'])
s1.replace('1234', '')
s1.replace('\d', '', regex = True)              # regex = True 설정 시 
                                                # 정규표현식 패턴을 갖는 문자열 치환

#[ 연습 문제 ]
# 1. emp.csv 파일을 읽고 
emp = pd.read_csv('data/emp.csv')

# 1) S로 시작하는 이름을 모두 NA로 변경
emp['ENAME'].replace('^S.+', NA, regex = True)

# 2) job에서 CLERK와 SALESMAN을 ANALYST로 변경
emp['JOB'].replace('^[CS].+', 'ANALYST', regex = True) # 정규표현식
emp['JOB'].replace('CLERK|SALESMAN', 'ANALYST', regex = True)
emp['JOB'].replace(['CLERK', 'SALESMAN'], 'ANALYST')   # 복수 문자열 리스트로 전달

# [ 참고 ] 특정 컬럼 or 객체의 값을 |로 전달할 경우 동시에 모두 치환 가능
p1 = Series(emp['JOB'].unique()).str.cat(sep = '|')
emp['JOB'].replace(p1,'ANALYST', regex = True)

# 3) DEPTNO에서 10을 인사부, 20을 총무부, 30을 재무부로 변경
emp['DEPTNO'].replace(10, '인사부').replace(20, '총무, '재무부'부').replace(30, '재무부')
emp['DEPTNO'].replace([10, 20, 30], ['인사부', '총무부'])
emp['DEPTNO'].replace({10 : '인사부', 20 : '총무부', 30 : '재무부'})





# =============================================================================
# 정규표현식 추출 함수
# =============================================================================
import re
dir(re)                                         # 정규표현식과 관련된 method 목록

# 1. findall
#    정규표현식 혹은 특정 패턴에 만족하는 문자열 모두 추출
#    일반적으로 벡터 연산 불가
#    str.findall 사용시 벡터 연산 가능


# 정규표현식 compile : 파싱 과정 생략으로 cost 감소
re.compile(pattern = ,                          # 파싱 대상 정규표현식
           flags = )                            # 옵션(ex. 대소구분 무시 등)


# 예제) 아래 문자열에서 이메일 주소 추출
vstr1 = 'abc !@~ 가나12 a1234@naver.com fje 12 aaa_12@google.com dkjf'
r1 = re.compile('.+@.+', re.IGNORECASE)
r2 = re.compile('[a-z0-9_]+@[a-z]+\.[a-z]+', re.IGNORECASE)
r3 = re.compile('[a-z0-9_]+@[a-z.]+', re.IGNORECASE)

re.findall(r1, vstr1)                           # 함수 형태 전달 가능 

r1.findall(vstr1)                               # 정규표현식에서 호출 가능
r2.findall(vstr1)                               # 정규표현식에서 호출 가능
r3.findall(vstr1)                               # 정규표현식에서 호출 가능


# 2) 시리즈에 패턴 추출
s1= Series(['dfd dlk a12@nvar.com', 'df', '3r49! A_12@goggle.com'])

s1.map(lambda x : r3.findall(x)).str[0]         # mapping 처리
s1.str.findall(r3).str[0]                       # str.findall



# 3) 그룹 추출
# () : 문자열에서 찾을 패턴을 전달하고 해당 패턴 내 추출 대상 정의
r3 = re.compile('([a-z0-9_]+)@([a-z.]+)', re.IGNORECASE)
s1.str.findall(r3).str[0].str[0]




# [ 참고 - 정규식 표현식에 따른 추출 과정 ]
vstr2 = 'a12!@#$%^&*() _+한글'
r1 = re.compile('\d')
r1.findall(vstr2)       # 숫자만 추출

r1 = re.compile('\D')   
r1.findall(vstr2)       # 숫자가 아닌것 추출(공백포함)

r1 = re.compile('\w')
r1.findall(vstr2)       # 문자(영문,한글) 또는 숫자 또는 _ 추출

r1 = re.compile('\W')
r1.findall(vstr2)       # _를 제외한 특수기호 추출(공백포함)




# 2. search
#    정규표현식을 포함한 패턴에 매칭되는 문자열 하나 추출(가장 먼저 발견되는)
#    위치 상관 없음(중간에 있는 문자열도 추출 가능)
#    group method로 원하는 문자열 추출 작업 필요

ss = 'lololo'
re.search('ol', ss)                             # ol과 일치하는 위치값 출력
re.search('ol', ss).group()                     # 'ol' 출력




# 3. match
#    정규표현식을 포함한 패턴에 매칭되는 문자열 하나 추출(가장 먼저 발견되는)
#    위치 중요(시작하는 문자열일 경우만 추출 가능)
#    group method로 원하는 문자열 추출 작업 필요

re.match('ol', ss)                              # None(아무것도 추출되지 않음)
re.match('lo', ss).group()





#[ 연습 문제 ]
# 교습현황.csv 파일에서 구, 동을 추출하여 구별, 동별, 교습과정별 총 합 출력
df1 = pd.read_csv('data/교습현황.csv', encoding = 'cp949', skiprows = 1)
df1 = df1.drop(['분야구분', '교습계열', '교습소명'], axis = 1)
df1.columns
df1.iloc[:, 3:] = df1.iloc[:, 3:].applymap(lambda x : int(x.replace(',', '')))
df1['총매출'] = df1.iloc[:, 3:].sum(axis = 1)
# 구, 동 추출
r1 = re.compile('.+\((.+)\)')
df1['구'] = df1['교습소주소'].str.findall('[가-힣]+구').str[0]
df1['동'] = df1['교습소주소'].str.findall(r1).str[0].str.findall('[가-힣]+동').str[0]
df1 = df1.drop(['교습소주소'], axis = 1)
# 총 합 출력
df1.groupby(['구', '동', '교습과정'])['총매출'].sum()


s2 = Series(['a_1', 'b!', 'C?', 'd-'])



#[ 연습 문제 ]
# ncs학원검색.txt 파일을 읽고
df1 = pd.read_csv('data/ncs학원검색.txt', encoding = 'cp949', sep = '|', header = None)

# 데이터 프레임 형식으로 변환
#       NAME           ADDRESS        TEL           START           END
#      아이티윌       서울 강남구    02-6255-8002    2018-10-12     2019-03-27

# 데이터 포함 행 추출
df1 = df1.iloc[:, 0].str.findall('.+훈련기간.+').str[0]
df1 = df1.dropna()

# 각 정보 추출
df2 = df2.drop(0, axis = 1)
df2['NAME'] = DataFrame(df1.str.findall('.+\(').str[0].str[:-4])
df2['ADDRESS'] = df1.str.findall('서울.+☎').str[0].str[:6]
df2['TEL'] = df1.str.findall('☎.+\)').str[0].str[2:-2]
df2['START'] = df1.str.findall('기간.+').str[0].str[5:15]
df2['END'] = df1.str.findall('기간.+').str[0].str[-12:]


# 한번의 scan으로 원하는 형식으로 데이터를 전달
import re
r1 = re.compile('(.+) \( ([가-힣 ]+) ☎ ([0-9-]+) \) .+ : ([0-9-]+) ~ ([0-9-]+)')

vre = df1.iloc[:, 0].str.findall(r1).str[0].dropna()
vname = vre.str[0].str.strip()
vaddr= vre.str[1].str.strip()
vtel = vre.str[2].str.strip()
vstart = vre.str[3].str.strip()
vend = vre.str[4].str.strip()

DataFrame({'NAME':vname, 'ADDR':vaddr, 'TEL':vtel, 'START':vstart, 'END':vend})




# =============================================================================
# 문자열에 대한 그룹 생성 및 추출
# =============================================================================
vstr = 'aa a12345@naver.com df1j1 ldf bcd@gmail.com dljf1'

# 이메일 전체 추출(그룹 없이)
r1 = re.compile('[a-z0-9]+@[a-z]+\.[a-z]+')
re.findall(r1,vstr)
re.search(r1, vstr).group()

# 이메일 구성 요서 각각 추출(그룹 생성)
r2 = re.compile('([a-z0-9]+)@([a-z]+)\.([a-z]+)')

# 1) findall
#    색인을 통해 그룹추출 가능
re.findall(re, vstr)
Series(re.findall(r2, vstr)).str[0]             # 색인을 통해 그룹 추출 가능
re.search(r2, vstr).group()                     # 패턴 매칭 전체 문자열 추출

[i.group() for i in re.finditer(r2, vstr)]      # 패턴 매칭 전체 문자열 추출
[i.group(1) for i in re.finditer(r2, vstr)]     # 1: 첫번째 그룹 
[i.group(2) for i in re.finditer(r2, vstr)]     # 2: 두번째 그룹 


re.search(r2, vstr).group(0)                    # 0: 전체 문자열

re.search(r2, vstr).group(1)                    # 1: 첫번째 그룹
re.search(r2, vstr).group(2)                    # 2: 두번째 그룹
re.search(r2, vstr).group(3)                    # 3: 세번째 그룹 
 