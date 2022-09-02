# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 벡터화가 내장된 string method
# =============================================================================
# 기본 Python 제공 string method
# pandas 제공 string method는 벡터연산 가능(Series만 가능 - DataFrame 적용 불가)
# str 호출 후 사용

dir(Series.str)                         # 벡터화가 내장된 string method 목록

# [ 참고 ] replace 형태 3가지
# 1) 기본 string method의 replace : 벡터연산 불가
# 2) 값 치환 method
# 3) pandas string method : 벡터연산 가능



# 1. replace
s1 = Series(['1,100', '2,200'])
s1.str.replace(',', '')


#[ 연습문제 ]
# card_history.csv 파일을 읽고 천단위 구분기호 삭제
card = pd.read_csv('data/card_history.csv', encoding = 'cp949')

# 1) 행별, 열별 fetch > apply + str.replace
f1 = lambda x : x.astype('str').str.replace(',', '')
card.apply(f1, axis = 0)
# 2) 원소별 fetch > applymap + replace
card.applymap(lambda x : str(x).replace(',', ''))



# 2. split
s2 = Series(['a/b/c', 'A/B/C'])
s2.str.split('/'



# 3. 색인
s2.str[2]
s2.str.get(2)

# 예시) student.csv 파일의 주민번호 컬럼에서 성별에 해당하는 숫자 추출
std = pd.read_csv('data/student.csv', encoding = 'cp949')

# 기본 색인 방식 str[n]
std['JUMIN'].astype('str').str[6]    
# 색인 함수 str.get(n)
std['JUMIN'].astype('str').str.get(6)



# 4. 문자열 개수 count
s2.str.count('a')



# [ 연습 문제 ]
# professor.csv 파일을 읽고
pro = pd.read_csv('data/professor.csv', encoding = 'cp949')

# 1) email_id 출력
pro['EMAIL'].str.split('@').str[0]

# 2) 입사연도 추출
pro['HIREDATE'].str[:4]

# 3) ID의 두번째 값이 a인 직원 출력
pro[pro['ID'].str[1] == 'a']



# 5. 문자열 결합(cat)
#    기본 string method 제공 X
s1.str.cat()                            # 시리즈의 모든 원소 결합
s1.str.cat(sep = '/')                   # 결합 시 분리구분 기호 전달 가능



# 6. 문자열 결합(join)
s3 = s2.str.split('/')
s3.astype('str').str.cat()              # 시리즈 내 리스트 간 결합
s3.str.join(sep = '')                   # 시리즈 내 리스트의 구성 원소 간 결합



# 7. 특정 string 포함 여부 확인
#    기본 string method 제공 X

pro['ID'].map(lambda x : 'a' in x)
pro['ID'].str.contains('a')
pro['ID'].str.count('a') >= 1



# 8. 반복
s1 * 3
s1.str.repeat(3)



# 9. 삽입
# 1) ljust, rjust (기본 method, str method)
s1.str.ljust(6, '0')                    # s1 문자열이 왼쪽 배치, 나머지 공간 '0'
s1.str.rjust(6, '0')                    # s1 문자열이 오른쪽 배치, 나머지 공간 '0'



# 2) zfill (기본 method, str method)
'a'.zfill(5)                            # 왼쪽에 전달된 너비만큼 0 채우기
s1.str.zfill(6)


# 3) pad (str method)
s1.str.pad(width, side = 'left', fillchar = ' ')
s1.str.pad(6, fillchar = '0')


# 4) 형식 변환
'%02d' % 5



# [ 연습 문제 ]
# professor.csv 파일을 읽고
pro = pd.read_csv('data/professor.csv', encoding = 'cp949')
# 1)  이름을 다음과 같은 형식으로 변경
# 서재수 > 서 재 수
pro['NAME'].str.join(sep = ' ')  

# 2) 홈페이지가 없는 사람의 홈페이지 주소를 새로 부여
#    http://www.itwill.com/email_Id
hpage = 'http://www.itwill.com/' + pro['EMAIL'].str.split('@').str[0]
pro['HPAGE'].fillna(hpage)



# 10. 위치
pro['EMAIL'].str.find('@')              # 없으면 -1 리턴



# 11. 문자열 길이
len(pro['ID'])                          # 시리즈 원소의 개수
pro['ID'].str.len()                     # 각 문자열의 길이



# 12. 공백제거(문자열제거)
s4 = Series([' a ', ' b ', 'a b'])
s4.str.strip().str.len()                # 양쪽 공백 제거
s4.str.strip('a')                       # 양쪽에서 특정 문자열 제거

s4.str.lstrip()                         # 왼쪽 공백 제거
s4.str.lstrip('a')                      # 왼쪽에서 특정 문자열 제거

s4.str.rstrip()                         # 오른쪽 공백 제거
s4.str.rstrip('b')                      # 오른쪽 특정 문자열 제거



# 13. 대소치환
s2.str.upper()
s2.str.lower()
s2.str.title()




# [ 연습 문제 ]
# 교습현황.csv 파일을 읽고
df1 = pd.read_csv('data/교습현황.csv', encoding = 'cp949', skiprows = 1)

# data cleansing
df1_s = Series(df1.columns)
df1_s[5:] = df1_s[5:].str[:4]
df1.columns = df1_s
df1 = df1.applymap(lambda x : x.replace(',', ''))
df1.iloc[:, 5:] = df1.iloc[:, 5:].astype('float')

# create a column '구'
df1['구'] = df1['교습소주소'].str[6:9]
df1['구'].unique()
# delete unused columns
df1 = df1.drop(['교습소주소', '분야구분', '교습계열'], axis = 1)


# 1) 구별 교습과정별 교습소별 연도별 총 합 출력
df1 = df1.groupby(['구', '교습과정', '교습소명']).sum()
df1 = df1.stack().reset_index()
df1 = df1.rename({'level_3':'년도', 0:'매출'}, axis = 1)
df1.groupby(['구', '교습과정', '교습소명', '년도'], as_index = False)['매출'].sum()


# 2) 구별 교습과정별 가장 인기있는 교습소명 출력
df1_sum = df1.groupby(['구', '교습과정', '교습소명'])['매출'].sum()
df1_sum[df1_sum.groupby(['구', '교습과정']).idxmax()]



