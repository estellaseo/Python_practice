# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# multi-level index
# =============================================================================
# index, column을 여러 level로 표현 가능
# 엑셀의 셀 병합을 표현하는 방법
# level의 이름을 전달하여 특정 레벨에 대한 연산, 그룹핑, 정렬 가능
# level의 위치(level 순서: 0, 1, 2, ...)

# 1. 생성
df1.index = ['객체1', '객체2']
df1.set_index(['col1', 'col2'])

df1.columns = ['객체1', '객체2']

# 예시) 외부객체를 활용한 multi-index
df1 = DataFrame(np.arange(1, 17).reshape(4, 4))
df1.index = [['A', 'A', 'B', 'B'], ['a', 'b', 'a', 'b']]
df1.columns = [[2007, 2007, 2008, 2008], [1, 2, 1, 2]]



# [ 연습 문제 ]
df2 = pd.read_csv('data/multi_index.csv', encoding = 'cp949')
df2.dtypes

# step 1) 지역컬럼 NA 이전값으로 치환
df2['Unnamed: 0'] = df2['Unnamed: 0'].fillna(method = 'ffill')[1:]
# step 2) 1st, 2nd 컬럼 index 전달
df2 = df2.set_index(['Unnamed: 0', 'Unnamed: 1'])
# step 3) Unnamed 포함하는 경우 NA 치환
df2.columns = df2.columns.map(lambda x : NA if 'Unnamed' in x else x)

#[ 참고 ] 정규표현식을 사용하여 치환할 값을 형태로 전달
Series(df2.columns).replace('^Unnamed.', NA, regex = True)

# step 4) 컬럼에서 NA 이전값 치환
# 주의: index object가 호출하는 fillna에 method 옵션 사용 불가
df2.columns = Series(df2.columns).fillna(method = 'ffill')
# step 5) 현 컬럼과 첫번째 행 column 전달
df2.columns = [df2.columns, df2.iloc[0, :]]
# step 6) 첫번째 행 제거
df2 = df2.drop(NA, axis = 0, level = 0)  # level=0의 index가 NA인 행(axis=0) 삭제
df2.iloc[1:, :]



# [ 참고 ] index names
df2.index.names                          # 복수형: multi-index의 이름 선택/수정
df2.columns.names = [None, None]         # column names 선택/수정

df2.index.names = ['지역', '지점']        # index names 변경
df2.columns.names = ['상품', '구분']      # 이름으로 지정 가능(ex. level = '구분')



# 2. 색인
df1[2007]                                # 상위 레벨에 대한 색인 방식(기존과 동일)
df1.loc['A', :]                          # 상위 레벨에 대한 색인 방식(기존과 동일)

df1.loc['a', :]                          # 하위 레벨에 대한 직접 색인 불가
df1[1]                                   # key indexing 불가


df1.iloc[:, (0, 2)]                      # 위치값 이용한 직접 색인

# xs multi-index의 하위 레벨에 직접 접근이 가능한 색인 메서드
df1.xs('a', axis = 0, level = 1)         # 1 level index로 리턴(차원 축소)
df1.xs('a', axis = 0, level = 1, drop_level = False)        # (차원 유지)

df1.index                                # index 저장 형태 : 튜플
df1.xs(('A', 'b'), axis = 0)             # 튜플로 순차적 접근 가능
df1.loc[('A', 'b'), :]                   # 튜플을 이용한 loc 색인 가능
df1.loc[[('A', 'b'), ('B', 'b')], :]     # 복수 색인 가능



# [ 연습 문제 ]
# 1. df2에서 price 삭제
df2.drop('price', axis = 1, level = 1)
# 2. A 지점만 선택
df2.xs('A', axis = 0, level = 1, drop_level = False)
# 3. 냉장고만 선택
df2.xs('냉장고', axis = 1, level = 0, drop_level = False)
# 4. 냉장고, qty 선택
df2.iloc[:, 1:2]
df2.xs(('냉장고', 'qty'), axis= 1, drop_level = False)
# 5. seoul, A와 incheon, B 선택
df2.loc[[('seoul', 'A'), ('incheon', 'B')], :]



#[ 연습 문제 ]
# multi_index_ex1.csv 파일을 읽고 multi-index 설정
df3 = pd.read_csv('data/multi_index_ex1.csv', encoding = 'cp949')
# index 설정
df3 = df3.set_index(['지역', '지역.1'])
# columns 설정
df3.columns = df3.columns.map(lambda x : x[:2])
df3.columns = [df3.columns, df3.iloc[0, :]]
# 1st 행삭제
df3 = df3.drop('지점', axis = 0, level = 1)
# 행이름, 컬럼명 삭제
df3.index.names = [None, None]
df3.columns.names = [None, None]


# 냉장고의 A지점 판매량 확인 > index, clumn 방향을 xs로 동시 전달 불가
df3.xs('냉장고', axis = 0, level = 1.xs('A'), asix = 1, level = 1)




# =============================================================================
# 3 level multi-index 생성 및 색인
# =============================================================================
df4 = DataFrame(np.arange(1,33).reshape(4,8))
df4.columns = [['A','A','A','A','B','B','B','B'], 
               ['서울','서울','경기','경기','서울','서울','경기','경기'],
               [2021,2022,2021,2022,2021,2022,2021,2022]]

# 2021년 데이터 선택
df4.xs(2021, axis = 1, level = 2)
df4.xs(('A', '서울', 2021), axis = 1, drop_level = False)

df4.xs(('A', '서울'), axis = 1)           # 상위 level부터 순차적 접근 가능

df4.xs(('서울', 2021), axis = 1)          # 중간 level부터 접근 불가
df4.xs(('서울', 2021), axis = 1, level = [1, 2]) # level 반드시 전달




# =============================================================================
# multi_index 산술 연산
# =============================================================================
df3 = df3.astype('int')

df3.sum(axis = 1)                # 가로방향 연산(각 행별 총 합 리턴)
df3.sum(axis = 1, level = 0)     # 컬럼 내 첫번째 level이 같은 값끼리 계신
                                 # (향후 level 사용 불가 예정)
                                 
# groupby 사용하여 동일 level값끼리 연산 가능
df3.groupby(axis = 1, level = 0).sum()




# =============================================================================
# multi-index 정렬
# =============================================================================

# 예시) df3에서의 첫번째 값 순서대로 행 재배치(정렬)
df3.sort_values(('서울', 'A'), ascending = False) # 특정 컬럼 tuple로 전달 및 정렬

df3.sort_values(('가전', 'TV'), axis = 1)         # 특정 행 tuple로 전달 및 정렬



# 2) index 순서 변경
df3.sort_index(axis = 1, level = 1)               # 2 level의 컬럼 순서 정렬
# multi-index에서의 level 순서변경
df3.swaplevel(0, 1, axis = 1)                      # 컬럼 정렬 없이 순서 변경

# 컬럼 정렬 후 순서 변경
df3.sort_index(axis = 1, level = 1).swaplevel(0, 1, axis = 1) 



#예시) multi-index를 갖는 경우 최댓값 찾기
# delivery에서 시간대별로 배달건수가 가장 많은 업종 찾기
deli = pd.read_csv('data/delivery.csv', encoding = 'cp949')

deli2 = deli.groupby(['시간대', '업종'])['통화건수'].sum()
# 시간대별 max 통화건수를 갖는 업종 tuple 형태(multi-index)로 가능
deli2.groupby(level = 0).idxmax()
# 통화건수와 함께 출력
deli2[deli2.groupby(level = 0).idxmax()]




