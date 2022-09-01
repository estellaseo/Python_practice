# -*- coding: utf-8 -*-
run my_profile

# taxi_call.csv 파일을 읽고
taxi = pd.read_csv('data/taxi_call.csv', encoding='cp949')

# 1. 발신지_시군구별로 택시 콜이 가장 많은 발신지_읍면동
# long data, multi-index X
taxi_sum = taxi.groupby(['발신지_시군구', '발신지_읍면동'], 
                        as_index = False)['통화건수'].sum()
idx1 = taxi_sum.groupby('발신지_시군구')['통화건수'].idxmax()
taxi_sum.iloc[idx1]
# long data, multi-index O
taxi_sum2 = taxi.groupby(['발신지_시군구', '발신지_읍면동'], 
                         axis = 0)['통화건수'].sum()
taxi_sum2[taxi_sum2.groupby('발신지_시군구').idxmax()]
# wide data
taxi_sum3 = taxi.pivot_table(index = '발신지_시군구', columns = '발신지_읍면동', 
                             values = '통화건수', aggfunc = 'sum')
taxi_sum3.idxmax(axis = 1)




# 2. 발신지_시군구별로 택시 콜이 가장 많은 시간대 top5
taxi_time = taxi.groupby(['발신지_시군구', '시간대'])['통화건수'].sum()

# rank(시군구별 순위별 정렬 및 출력되지 않는 현상 > 추가 정렬 필요)
taxi_rank = taxi_time.groupby('발신지_시군구').rank(ascending = False, method = 'first')
taxi_time[taxi_rank <= 5].reset_index().sort_values(['발신지_시군구', '통화건수'], ascending = [True, False])
f1 = lambda x : x.sort_values(ascending = False)
taxi_time[taxi_rank <= 5].groupby(level = 0, group_keys = False).apply(f1)

#정렬 후 큰 순서대로 5개 추출
f2 = lambda x : x.sort_values(ascending = False)[:5]
taxi_time.groupby('발신지_시군구', group_keys = False).apply(f2)

     

                                               
# 3. 요일별 시간대별 통화건수 총 합 출력(wide)
taxi.dtypes
taxi['요일'] = taxi['기준년월일'].map(lambda x : 
                               datetime.strptime(str(x), 
                                                 '%Y%m%d').strftime('%A'))    
taxi.groupby(['요일', 
              '시간대'])['통화건수'].sum().unstack().iloc[[1, -2, -1, -3, 0, 2, 3]]
taxi.pivot_table(index = '시간대', columns = '요일', values = '통화건수', aggfunc = 'sum').iloc[:, [1, -2, -1, -3, 0, 2, 3]]

