# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 연관성 분석
# =============================================================================
# - 비지도 학습
# - 장바구니 분석, 서열 분석이라고도 부름
# - 하나의 장바구니에서의 각 상품들끼리 연관성이 다른 장바구니에도 발견되는지 확인
#   : 추천, 번들상품, 상품진열 등에 사용


# 연관 강도 평가 지표
# 1. 지지도(Support)
#    - P(A ∩ B) : A와 B가 동시에 포함된 거래 수 / 전체 거래 수
#    - 전체 거래 중 항목 A와 항목 B를 동시에 포함하는 거래의 비율
#    - 연관성 분석의 초기 기준점 > 거래수 기반 지지(낮은 거래 비율은 지지 X)

# 2. 신뢰도(Confidence)
#    - P(B|A) = P(A ∩ B) / P(A) : A와 B가 동시에 포함된 거래수 / A를 포함하는 거래 수
#    - A 상품을 샀을 때 B 상품을 살 조건부 확률
#    - 추천 할 만 하다고 보여지는 기준(강도)

# 3. 향상도(Lift)
#    - A와 B가 동시 포함 거래수 * 전체 거래수 / A 포함 거래 수 * B 포함 거래 수
#    - 규칙이 우연에 의해 발생한 것인지 판단하기 위함(우연성 제거)
#    - 두 품목의 상관관계를 기준으로 도출된 규칙의 예측력을 평가하는 지표
#    - 향상도 = 1 : 서로 독립적 관계
#      향상도 > 1 : 양의 상관관계, 향상도 < 1 : 음의 상관관계(연관성이 강하지 X)




# 예시)
# A편의점의 최근 1년 장바구니는 총 500개로 확인되었다.
# 그 중 각각의 구매 횟수가 다음과 같다고 가정하자.

# 군고구마 총 구매 횟수 : 50
# 군고구마 우유 구매 횟수 : 25
# 우유 총 구매 횟수 : 50

# 삼각깁밥 총 구매 횟수 : 100
# 삼각김밥, 컵라면 구매 횟수 : 80
# 컵라면 총 구매 횟수 : 90

# 불닭볶음면 총 구매 횟수 : 5
# 불닭볶음면, 스트링 치즈 구매 횟수 : 4
# 스트링 치즈 총 구매 횟수 : 10

#                                지지도       신뢰도       향상도
# 군고구마 > 우유의 추천             0.05        0.5          5
# 삼각김밥 > 컵라면                 0.16        0.8         4.44
# 불닭볶음면 > 스트링 치즈 추천      0.008        0.8         




# 예제) 연관분석 실습
# 1. data loading
vlist = [['맥주', '오징어', '치즈'],
         ['소주', '맥주', '라면'],
         ['맥주', '오징어'],
         ['라면', '김치', '계란'],
         ['맥주', '소세지']]


# raw data
# 거래번호     상품이름       가격      거래일자
#   0001        맥주        ...
#   0001       오징어
#   0001        치즈


# 2. 연관분석 모듈(패키지) 설치
pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder   # 트랜젝션 데이터 변환 함수
from mlxtend.frequent_patterns import association_rules, apriori # 연관분석 수행


# 3. 트랜젝션 데이터 변환
m_trans = TransactionEncoder()
m_trans.fit(vlist)
alist = m_trans.transform(vlist)            # 트랜젝션 데이터 각 구매품목별 구매여부
df_result = DataFrame(alist, columns = m_trans.columns_)

# 4. 연관분석 학습
m_ap = apriori(df_result, min_support=0.1, use_colnames=True)
dt_total = association_rules(m_ap, metric = 'lift')

dt_total
dt_total.columns        # ['antecedents', 'consequents', 'antecedent support',
                        # 'consequent support', 'support', 'confidence', 
                        # 'lift', 'leverage', 'conviction']


dt_total.loc[:, ['antecedents', 'consequents', 'support', 'confidence', 'lift']]




# [ 연습 문제 ]
df_menu = pd.read_csv('data/chipotle.tsv', sep = '\t')
df_menu.columns
df_menu['item_name'].unique()

# 1) order_id 별 item_name 결합
df_menu.loc[:, ['order_id', 'item_name']]

vlist = []
for i in df_menu['order_id'].unique() :
    vlist.append(list(df_menu.loc[df_menu['order_id'] == i, 'item_name']))
    
# [ 참고 ] 시리즈 형식 목록 만들기
f1 = lambda x : x.str.cat(sep = ',').split(',')
df_menu.groupby('order_id')['item_name'].apply(f1)


# 2) 트랜젝션 데이터 변환
m_trans = TransactionEncoder()
m_trans.fit(vlist).transform(vlist)
df_result = DataFrame(alist, columns = m_trans.columns_)


# 3) 연관분석
m_ap = apriori(df_result, min_support=0.01, use_colnames=True)
dt_total = association_rules(m_ap, metric = 'lift')

df_total2 = dt_total.loc[:, ['antecedents', 'consequents', 'support', 'confidence', 'lift']]


# A, B를 동시에 주문할 비율 (support) XX% / A, C > B
dt_total2.loc[:, ['antecedents', 'consequents', 'support']]
df_total3 = df_total2.loc[df_total2['lift'] > 1, 
                          ['antecedents', 'consequents', 'support', 'confidence', 'lift']]

df_total3.sort_values(['confidence', 'lift'], ascending = False)
df_total3.to_csv('df_total_menu.csv', index = False)




