# -*- coding: utf-8 -*-
run my_profile

# 1. emp.csv 파일을 읽고
emp = pd.read_csv('data/emp.csv')

# 1) 행 이름을 각 사원의 사번으로 처리
emp = emp.set_index('EMPNO')
emp.index

# 2) 본문에서 사원번호 컬럼 제외
emp.index

# 3) NA를 0으로 수정
emp.fillna(0)

# 4) 컬럼 이름을 모두 소문자로 변경
emp.columns = emp.columns.map(lambda x : x.lower())

# 5) 7902 사번을 9000으로 수정
emp.rename({7902 : 9000}, axis = 0)




# 2. card_history.csv 파일을 읽고
card = pd.read_csv('data/card_history.csv', encoding = 'cp949')

# 각 일자별로 지출품목에 대한 지출 비율 출력
card = card.set_index('NUM')
card = card.applymap(lambda x : int(x.replace(',', '')))

f_ratio = lambda x : round(x / x.sum() * 100, 2)
card.apply(f_ratio, axis = 1)




# 3. subway2.csv  파일을 읽고
sub = pd.read_csv('data/subway2.csv', encoding = 'cp949', skiprows = 1)

# 1) 다음의 데이터 프레임 형식으로 변경(역이름 호선 정보 그대로)
# 전체       구분   5시       6시     7시 ...
# 서울역(1)  승차 17465  18434  50313 ...
# 서울역(1)  하차 ....

# 방법 1)
sub['전체'] = sub['전체'].fillna(method = 'ffill')
a1 = Series(np.arange(5, 25).astype('str')) + '시'
sub.columns = ['전체'] + ['구분'] + list(a1)
# 방법 2)
a1 = Series(sub.columns)
a1[2:] = a1[2:].map(lambda x : str(int(x[:2])) + '시')
sub.columns = a1


# 2) 역별 승차 총 합
sub2 = sub.loc[sub['구분'] == '승차', :].drop('구분', axis = 1).set_index('전체')
# (set_index('전체')를 함으로써 승차 총 합과 역이름이 함께 출력됨)
sub2.sum(axis = 1)
# 승차 총합 내림차순으로 정렬 후 승차 수 Top 5 역 추출
sub2.sum(axis = 1).sort_values(ascending = False)[:5]


# 3) 역별 승차가 가장 많은 시간대 출력
sub2.idxmax(axis = 1)           # 최대값에 해당하는 key 출력
