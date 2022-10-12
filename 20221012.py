# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 크롤링 
# =============================================================================
# [ website의 3가지 요소 ]
# 1. HTML(Hyper Text Markup Language) 
#    : www를 통해 볼 수 있는 문서를 만드는 언어

# 2. CSS(Casading Style Sheets) 
#    : HTML의 제약적인 부분을 보완하기 위해 만들어진 스타일 시트
#     (글꼴, 글자 크기, 줄 간격, 글 색, 배경 색 ..... 기타 등등)

# 3. javascript 
#    : 웹페이지의 동작을 담당하는 스크립트 언어



# [ 외부 데이터 가져오기 ]
# 1. 웹 스크래핑(Web Scraping) 
#   : 웹상에서 데이터를 가져오는 모든 행위 

# 2. 웹 크롤링(Web Crawling) 
#   : Web Scraping을 자동화하여 웹에서 데이터를 가져오는 행위

# 파이썬 크롤링 library
# - HTTP Client : Requests 
# - HTML / XML parser : BeautifulSoup(HTML / XML 파일을 분석해주는 라이브러리)



# [ TEST1 : 요소 접근 ]
# 1. 라이브러리 로딩
from bs4 import BeautifulSoup


# 2. HTML 소스 가져오기(추후 requests를 통해 직접 가져옴)
#    - p : 하나의 문단을 구성
#    - h1~h6 : 각 세션의 제목으로 h1이 가장 높은 등급

html = """
<html><body>
  <h1>웹크롤링 강의</h1>
  <p>문단입니다</p>     
  <p>문단입니다2</p>
</body></html>
"""


# 3. HTML 파싱(구조 분석) 
soup = BeautifulSoup(html, 'html.parser')


# 4. 정보 추출(사용자 정의)
#    - 요소의 순차적 접근으로 데이터 가져오기
h1 = soup.html.body.h1
h1.string

p1 = soup.html.body.p
p1.string

p2 = p1.next_sibling.next_sibling    
p2.string



# [ TEST2. find 메서드 사용 ]
# - find()     : 속성을 지정해서 원하는 정보를 찾는 메서드
# - find_all() : 여러 속성(태그)를 한 번에 추출 메서드

# 1. 라이브러리 로딩
from bs4 import BeautifulSoup


# 2. HTML 소스 가져오기
html = """
<html><body>
  <h1 id="title">웹크롤링 강의</h1>
  <p id="body">웹 스크래핑이란?</p>
  <p>배워볼까요?</p>
</body></html>
"""


# 3. HTML 파싱
soup = BeautifulSoup(html, 'html.parser')


# 4. 정보 추출
title = soup.find(id='title')
title.string

body = soup.find(id='body')
body.string



# [ TEST3. find_all 메서드 사용 ]
# 1. 라이브러리 로딩
from bs4 import BeautifulSoup

# 2. HTML 소스 가져오기
#    - ul   : unordered list
#    - ol   : ordered list
#    - li   : LIst의 약자로 ul, ol과 함께 사용하는 목록을 표현하는 요소
#    - href : url로 연결할 수 있는 하이퍼링크를 생성해주는 속성으로 주로 a 요소와 같이 사용


html = """
<html><body>
  <ul>
    <li><a href="http://www.naver.com">naver</a></li>
    <li><a href="http://www.daum.net">daum</a></li>
  </ul>
</body></html>
"""


# 3. HTML 파싱
soup = BeautifulSoup(html, 'html.parser')


# 4. 정보 추출**
links = soup.find_all('a')

for a in links :
    vurl = a.attrs['href']        # a 요소에 있는 href 속성 추출(attrs 대신 get 사용 가능)
    vtext = a.string              # a 요소에 있는 문자열 추출(string 대신 text 사용 가능)
    print(vtext, ':', vurl)

for a in links :
    vurl = a.get('href')        
    vtext = a.text             
    print(vtext, ':', vurl)



# [ TEST4. 선택자 사용하기 ] select, select_one 사용
# 선택자(요소를 전체적인 느낌으로 정의하는 방식) 기본 서식
# * : 모든 요소를 선택
# <요소명> : 특정 요소를 선택
# .<클래스이름> : 특정 클래스를 선택
# #<id이름> : 특정 속성 선택

# 선택자들 관계 지정
# <선택자> , <선택자> : 여러 선택자 모두 선택
# <선택자> <선택자> : 앞 선택자 하위에 있는 다음 선택자 지정
# <선택자> > <선택자> : 앞 선택자 하위에 있는 바로 다음 선택자 지정
# <선택자> + <선택자> : 같은 계층에서 바로 뒤에 오는 요소 지정
# <선택자1> ~ <선택자2> : 선택자1에서 선택자2의 모든 요소 지정

# 1. 라이브러리 로딩


# 2. html 가져오기
# - div : 특별한 의미는 없으며 문서의 영역을 지정하는데 사용 요소
# - id 속성 : 식별자
# - class 속성 : 정의를 내리기 위한 구분자    

html = """
<html><body>
<div id="meigen">
  <h1>위키북스 도서</h1>
  <ul class="items">
    <li>그레이스와 함께하는 파이썬 활용편</li>
    <li>그레이스와 함께하는 즐거운 생활 코딩</li>
    <li>그레이스와 함께하는 배드민턴의 세계</li>
  </ul>
</div>
<div id="meige2">
  <h1>위키북스 도서</h1>
  <ul class="items">
    <li>그레이스와 함께하는 파이썬 활용편</li>
    <li>그레이스와 함께하는 즐거운 생활 코딩</li>
    <li>그레이스와 함께하는 배드민턴의 세계</li>
  </ul>
</div>
</body></html>
"""

# 3. 파싱
soup = BeautifulSoup(html, 'html.parser')


# 4. 정보 추출
h1 = soup.select_one("div#meigen > h1").string
l1 = soup.select("div#meigen > ul.items > li")
l1 = soup.select("li")

for i in l1 :
    print(i.string)



# [ TEST5 : 위키 문헌의 윤동주 작가의 작품 목록 가져오기 ]
# 1. url 가져오기
from urllib.request import urlopen 

url = "https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%EC%9C%A4%EB%8F%99%EC%A3%BC"
html = urlopen(url)


# 2. html parsing
soup = BeautifulSoup(html, 'html.parser')


# 3. 작품 이름 가져오기 위한 CSS 선택자 확인
# 3-1)우클릭 하여 개발자 도구 확인  
# 3-2) 원하는 부분을 선택하여 html 요소 확인(우클릭 > 복사 > select)
#      => #mw-content-text > div.mw-parser-output > ul:nth-child(6) > li > b > a   


# 4. 추출
vlist = soup.select('a')
vlist = soup.select('div.mw-parser-output > ul > li a')

output = []
for i in vlast:
    


# [ TEST6 : 아디다스 운동화 목록 가져오기 ]
from urllib.request import urlopen 
from bs4 import BeautifulSoup

url = 'https://sneakernews.com/category/adidas'
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

vlist = soup.select('div.post-content > h4 > a')

outlist = []
for i in vlist :
    outlist.append(i.string)
    
len(outlist)


# 다른 페이지에 있는 정보 가져오기
url = 'https://sneakernews.com/category/adidas/page/2'
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

vlist = soup.select('div.post-content > h4 > a')

outlist = []
for i in vlist :
    outlist.append(i.string)
    
len(outlist)


# 여러 페이지에 있는 정보 가져오기(1~10page)
outlist = []
vlist = soup.select('div.post-content > h4 > a')

for page_num in range(1, 11) :
    url = 'https://sneakernews.com/category/adidas/page/' + str(page_num)
    
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    
    vname = [a_name.string for a_name in vlist]
    outlist.append(vname)
   
df_name = DataFrame(outlist, index = range(1, 11))
df_name = df_name.stack().reset_index()
df_name.drop('level_1', axis = 1)
df_name.columns = ['page_num', 'name']


# 모든 페이지에 있는 정보 가져오기(페이지 정부가 없는 경우 크롤링 stop)
from urllib.request import HTTPError

url = 'https://sneakernews.com/category/adidas/page/' + str(10000)
html = urlopen(url)

def f_page(index) :
    try :
        html = urlopen('https://sneakernews.com/category/adidas/page/' + str(index)) 
    except HTTPError as e :
        return None
    else :
        soup = BeautifulSoup(html, 'html.parser')
        vlist = soup.select('div.post-content > h4 > a')
        
        vname = [ i.text for i in vlist ]
        
        return vname  
    
    
def adidas_crw() :
    page_num = 1
    df_result = []
    while True :
        vname = f_page(page_num)
        if vname is None :
            break
        else : 
            df_result.append({'page':page_num, 'name':vname})
        page_num += 1            # page_num = page_num + 1
    return df_result

f_page(1100)

import time
start = time.time()
adidas_crw()
end = time.time()

print(end - start)

    

# [ TEST7 : 네이버 날씨 가져오기(특정 지역) ]
import requests
from bs4 import BeautifulSoup

# 1. url 확인하기
#    다음의 두 형태로 선택 가능
vaddr = '강릉'
vhtml = requests.get('https://search.naver.com/search.naver?query='+ vaddr +'날씨')
vhtml = requests.get('https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query='+ vaddr +'날씨')


# 2. error code 확인
# 200 : 정상
# 404 : Not Found
vhtml.status_code


vaddr = '평양'
vhtml = requests.get('https://search.naver.com/search.naver?query='+ vaddr +'날씨')
vhtml.status_code
# > 정상 출력과 비정상 출력을 status_code로는 구분 불가
#   어떤 형태로 두 값의 구분을 할 것인지 확인
#   span 요소의 유무 여부로 정상 출력 및 비정상 출력 구분 가능


vaddr = '강릉'
html = requests.get('https://search.naver.com/search.naver?query='+ vaddr +'날씨')
soup = BeautifulSoup(html.text, 'html.parser')
error_check = soup.find('span', {'class' : 'text'} )

if error_check is None :
    print('해당 지역 날씨는 제공되지 않습니다.')
else :
    print('해당 지역 날씨 정보입니다.')
    
# 온도 가져오기
vlist = soup.select('.temperature_text > strong')
vtemp = vlist[0].text[5:]
print('현재 온도 : ', vtemp)

# 맑음/흐림 정보 가져오기
vwther = soup.select('div.temperature_info > p > span.weather.before_slash')[0].string
print('현재 날씨 : ', vwther)

# 미세먼지, 초미세먼지, 자외선, 일몰 가져오기
vtime = soup.select('ul.today_chart_list > li > a > span.txt')[:4]
v1 = ['미세먼지', '초미세먼지', '자외선', '일몰']
for i, j in zip(v1, vtime) : 
    print(i, j.string)


# 함수 정의
def crawling_weather(addr) :
    vhtml = requests.get('https://search.naver.com/search.naver?query='+ addr +'날씨')
    soup = BeautifulSoup(vhtml.text, 'html.parser')
    error_check = soup.find('span', {'class':'celsius'})
    
    # URL ERROR CHECK
    if error_check is None :
        print('해당 지역 날씨는 제공되지 않습니다')
    else :
        print('해당 지역 날씨 정보입니다')
        
        # 온도 가져오기
        temp = soup.select('.temperature_text > strong')[0].text[5:]
        print('현재 온도',':',temp)
    
        # 흐림정보 가져오기
        weath = soup.select('div.temperature_info > p > span.weather.before_slash')[0].text
        print('현재 날씨',':',weath)
    
        # 미세먼지, 초미세먼지, 자외선, 일몰 가져오기
        total = soup.select('div.report_card_wrap > ul > li > a > span')[:4]
        text = ['미세먼지', '초미세먼지', '자외선', '일몰']
    
        for i,j in zip(text, total) :
            print(i,':', j.text)


crawling_weather('강릉')
crawling_weather('대구')
crawling_weather('서울')




# =============================================================================
# TEST8 : Open API를 활용한 정보 가져오기
# =============================================================================
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
# TEST8 : 교보문고 책
# =============================================================================
# 1. url 가져오기
