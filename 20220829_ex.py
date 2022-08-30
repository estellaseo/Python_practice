# -*- coding: utf-8 -*-
run my_profile


# 1. sales2.csv 파일을 읽고
sales = pd.read_csv('data/sales2.csv', encoding='cp949')

# 1) 날짜, 지점, 품목을 멀티 인덱스로 설정
sales = sales.set_index(['날짜', '지점', '품목'])

# 2) 출고 컬럼이 높은 순서대로 정렬
sales.sort_values(by = ['날짜', '지점', '출고'], 
                  ascending = [True, True, False])

# 3) 각 날짜와 지점 내의 품목을 가나다순으로 정렬
sales.sort_values(axis = 0, ['날짜', '지점', '품목'])
sales.sort_index(axis = 0, level = [0, 1, 2])

# 4) 품목 인덱스를 가장 상위 인덱스로 배치
# level을 리스트로 전달하여 복수 컬럼 정렬 후 swaplevel
sales.sort_index(axis = 0, level = [2, 1, 0]).swaplevel(2, 0)

# 5) 지점별 출고, 판매, 반품의 총 합 출력
sales.sum(axis = 0, level = 1)             # FutureWarning
sales.groupby(axis = 0, level = 1).sum()   # groupby를 이용하는 방법으로 대체 권고



# 2. 병원현황.csv 파일을 읽고
df1 = pd.read_csv('data/병원현황.csv', encoding='cp949', skiprows = 1)

# data cleansing
df1 = df1.drop(df1[df1['표시과목'] == '계'].index)
df1 = df1.drop(['항목', '단위'], axis = 1)

# 1) 멀티 인덱스 설정
df1 = df1.set_index(['시군구명칭', '표시과목'])
# 1-2) 멀티 컬럼 설정
a1 = Series(df1.columns)
df1.columns = [df1.columns.map(lambda x : str(x[:4])), 
               df1.columns.map(lambda x : str(x[-3:-2]))]

# 2) 분기를 오름차순 정렬(년도: 내림차순)
df1 = df1.sort_index(axis = 1, level = [0, 1], ascending = [False, True])

# 3) 표시과목과 지역 순서 변경
df1 = df1.sort_index(axis = 0, level = [1, 0]).swaplevel(0, 1, axis = 0)

# 4) 각 표시과목별 연도별 병원수의 총 합
df1_sum = df1.groupby(axis = 1, 
                      level = 0).sum().groupby(axis = 0, level = 0).sum()

# 5) 연도별 병원수가 가장 많은 표시과목 출력
# 연도별 총 합 기준
df1_sum.idxmax(axis = 0)
# 연도별 4분기 기준
df1.groupby(axis = 0, level = 0).sum().xs('4', axis = 1, level = 1).idxmax()

