# -*- coding: utf-8 -*-
run my_profile
import math

# 1. ex_test1.csv 파일을 읽고
df1 = pd.read_csv('data/ex_test1.csv', encoding='cp949')

# 1) 각 구매마다의 포인트를 확인하고 point 컬럼 생성
# point는 주문금액 50000 미만 1%, 5만 이상 10만 미만 2%, 10만 이상 3%
# sol 1) np.where(Series에 대한 조건 치환 적용)
df1['포인트'] = np.where(df1['주문금액'] < 50000, df1['주문금액'] * 0.01, 
                      np.where(df1['주문금액'] < 100000, df1['주문금액'] * 0.02, 
                               df1['주문금액'] * 0.03))
# sol 2) for문 + if문
point = []
for i in df1['주문금액'] :
    if i < 50000 :
        point.append(i * 0.01)
    elif i < 100000 :
        point.append(i * 0.02)
    else :
        point.append(i * 0.03)
df1['포인트'] = point
# sol 3) map + 사용자정의함수
def f1(x) :
    if x < 50000 :
        return (x * 0.01)
    elif x < 100000 :
        return (x * 0.02)
    else :
        return (x * 0.03)
df1['주문금액'].map(f1)


# 2) 회원번호별 총 주문금액과 총 포인트 금액 확인
df1_sum = df1.groupby('회원번호')['주문금액'].sum()

# 3) 회원별 주문금액을 확인하고 총 주문금액 기준 상위 30% 회원 번호의 총 합
Series(df1_sum.sort_values(ascending = False)[:math.trunc(len(df1_sum)*0.3)].index).sum()

# [ 참고 ] np.trunc
np.trunc(len(df1_sum)*0.3)



# 2. subway2.csv 파일을 읽고 
sub = pd.read_csv('data/subway2.csv', encoding='cp949', skiprows = 1)

# data cleasing
sub['전체'] = sub['전체'].fillna(method = 'ffill')
sub['전체'] = sub['전체'].str.split('(').str[0]
sub = sub.set_index(['전체', '구분'])
sub.columns = sub.columns.str[:2].astype('int')

# 1) 각 역별 승하차의 오전/오후별 인원수를 출력(24~01은 오전)
sub_sum = sub.stack().reset_index()
sub_sum = sub_sum.rename({'level_2':'시간대', 0:'인원'}, axis = 1)
sub_sum['오전오후'] = sub_sum['시간대'].map(lambda x : '오전' if (x  < 13)|(x == 24) else '오후')
sub_sum.groupby(['전체', '구분', '오전오후'])['인원'].sum()

# [ 해설 ]
g1 = sub.columns.map(lambda x : '오전' if (x  < 13)|(x == 24) else '오후')
sub.groupby(g1, axis = 1).sum()



# 2) 각 시간대별 승차인원이 가장 큰 5개의 역이름과 승차인원을 함께 출력
sub_on = sub_sum[sub_sum['구분'] == '승차'].drop('구분', axis = 1)
sub_on2 = sub_on.groupby(['시간대', '전체'])['인원'].sum()
f1 = lambda x : x.sort_values(ascending = False)[:5]
sub_on2.groupby('시간대', group_keys = False).apply(f1)

# [ 해설 ] wide data일 때 역 이름 출력
sub1 = sub.xs('승차', level = 1)
f1 = lambda x : x.sort_values(ascending = False)[:5].index
sub1.apply(f1, axis = 0)


