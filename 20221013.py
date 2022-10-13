# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame

# =============================================================================
# Open API를 통한 크롤링 (json)
# =============================================================================
# json : 정형 데이터로 표현할 수 있는 데이터를 나열형식(key-value)으로 제공(반정형)

# 기본제공 url : http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/
# 추가 url
# getHoliDeInfo    	국경일 정보조회
# getRestDeInfo   	공휴일 정보조회
# getAnniversaryInfo	기념일 정보조회
# get24DivisionsInfo	24절기 정보조회
# getSundryDayInfo	잡절 정보조회


# 1. url open
import requests
url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
api_key_utf8 = 'KiowcYn0T5pJHycrZhkhZfqvqlWrk8eCP6nIRX1YUZf5TIQLLG0kh%2F6PYHwXLRbmYziJXniO76smUIwIpoKd%2Bg%3D%3D'
api_key_decoding = requests.utils.unquote(api_key_utf8)



# 2. parameter 정의
params = {
    'ServiceKey' : api_key_decoding,     # decoding key 전달
    'pageNo' : '1',
    'numOfRows' : 30,
    'solYear' : year,
    '_type' : 'json'
    }


   
# 3. 데이터 요청(requests or httpx 사용)
# pip install httpx
import httpx
response = requests.get(url, params = params)
response = httpx.get(url, params = params, timeout = 5)

response.status_code

contents = response.json()

dict_data = contents['response']['body']['items']['item']
DataFrame(dict_data)



# 4. 크롤링 함수 생성
from datetime import datetime

def api_get(year: datetime) -> pd.DataFrame:
    decoding = ''
    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'

    params = {'serviceKey': decoding,
              'pageNo': '1',
              'numOfRows': 30,
              'solYear': year,
              '_type': 'json'    # 데이터 불러올 형식 json or xml
              }

    is_end: bool = False
    holidays = []

    while is_end is not True:

        res = httpx.get(url, params=params, timeout=5)

        if res.status_code >= 300:
            raise httpx.RequestError(f"Error response {res.status_code} while requesting")

        contents = res.json()

        partial = []
        for data in contents["response"]["body"]["items"]["item"]:
            partial.append(data)
        holidays += partial

        # 공휴일 개수가 params에서 설정한 "numOfRows" 개수 보다 작으면 return, 
        # 아닐 경우 params의 pageNo을 1개씩 추가
        if len(partial) < 30:
            is_end = True
        else:
            params["pageNo"] += 1

    return pd.DataFrame(holidays)


api_get(2023)




# =============================================================================
# Open API를 통한 크롤링 (xml)
# =============================================================================
# xml : 확장형 마크업 언어(html 개선 언어 > 구조화)


# 예제) 해양 수산물 수입가격 가져오기 (xml)
# 데이터 설명 : https://www.data.go.kr/data/15056556/openapi.do
# 데이터 요청 url : http://apis.data.go.kr/1192000/select0070List/getselect0070List


# 1. 필요 모듈 호출
import requests
import httpx
from bs4 import BeautifulSoup



# 2. url 정의
url = 'http://apis.data.go.kr/1192000/select0070List/getselect0070List'
api_key_utf8 = 'KiowcYn0T5pJHycrZhkhZfqvqlWrk8eCP6nIRX1YUZf5TIQLLG0kh%2F6PYHwXLRbmYziJXniO76smUIwIpoKd%2Bg%3D%3D'
api_key_decoding = 'KiowcYn0T5pJHycrZhkhZfqvqlWrk8eCP6nIRX1YUZf5TIQLLG0kh/6PYHwXLRbmYziJXniO76smUIwIpoKd+g=='
# api_key_decoding = requests.utils.unquote(api_key_utf8)

params = {'serviceKey' : api_key_decoding, 
          'numOfRows' : '10', 
          'pageNo' : '1', 
          'type' : 'xml', 
          'baseDt' : '202201', 
          'imxprtSeNm' : '수입', 
          'mprcExipitmNm' : '연어' 
          }



# 3. 데이터 불러오기
from urllib.request import urlopen
urlopen?

res = requests.get(url, params = params)
content = res.text
xml_obj = BeautifulSoup(content, 'lxml-xml')
rows = xml_obj.findAll('item')
print(rows)

len(rows)
rows[0]                                  # 각 key(column)를 html 요소처럼 표현
rows[0].find_all()                       # 모든 요소 리턴



# 4. xml > DataFrame 변환
def f_to_dataframe(rows) :
    row_list = [] ; name_list = [] ; value_list = []

    for i in range(0, len(rows)):
        columns = rows[i].find_all()
        
        # 각 요소명을 컬럼명으로 사용하기 위해 첫번째 행에 대해서 요소명 추출
        for j in range(0,len(columns)):
            if i ==0:
                name_list.append(columns[j].name)  # name_list : 컬럼명
                
            # 각 행의 값 추출
            value_list.append(columns[j].text)     # value_list : 각 행의 값
      
        # 모든 행에 대한 값 추출
        row_list.append(value_list)                # row_list : 모든 행의 값
        value_list=[]                              # value_list 초기화
     
    df = pd.DataFrame(row_list, columns=name_list)
    return df


f_to_dataframe(rows)




