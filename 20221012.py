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



# [ 크롤링 과정 ]
# 1. url 정의

# 2. url에 해당되는 html 파일 가져오기
#    1) urlopen
#       - from urllib.request import urlopen 선언 후 urlopen 함수로 url 호출
#       - 호출된 html 파일을 BeautifulSoup으로 파싱
#       - 정의되지 않은 url에 대해 즉각적인 HTTP error 발생

#    2) requests
#       - import requests 선언 후 get 메서드로 url 호출
#       - get/post 가능(데이터 가져오기/전달하기)
#       - 딕셔너리 형식으로 데이터 전달 > 데이터의 세부속성 전달이 수월(params)
#       - 정의되지 않은 url에 대해 즉각적인 HTTP error 발생 X
#       - API, HTTP/1.1만 지원

#    3) httpx (설치 필요)
#       - 설치 후 import httpx 선언하여 get 메서드로 url 호출
#       - get/post 가능(데이터 가져오기/전달하기)
#       - API, HTTP/1.1, HTTP/2. 지원
#       - requests 사용불가일 경우 유용함

#    4) selenium (설치 필요)
#       - 설치 후 import selenium 선언하여 사용
#       - 웹사이트에 동적 호출 전달의 목적(텍스트 입력, 로그인, 버튼 클릭 등)


# 3. 소스 분석 후 원하는 데이터 가져오기
#    1) html
#       - find : 특정 요소로 직접 접근
#       - select : 요소 관계(선택자; selector) 정의
#    2) json
#    3) xml




# =============================================================================
# TEST 1. 요소 접근
# =============================================================================
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




# =============================================================================
# TEST 2. find 메서드 사용
# =============================================================================
# find()     : 속성을 지정해서 원하는 정보를 찾는 메서드
# find_all() : 여러 속성(태그)를 한 번에 추출 메서드

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




# =============================================================================
# TEST 3. find_all 메서드 사용
# =============================================================================
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


# 4. 정보 추출
links = soup.find_all('a')

for a in links :
    vurl = a.attrs['href']     # a 요소에 있는 href 속성 추출(attrs 대신 get 사용 가능)
    vtext = a.string           # a 요소에 있는 문자열 추출(string 대신 text 사용 가능)
    print(vtext, ':', vurl)

for a in links :
    vurl = a.get('href')        
    vtext = a.text             
    print(vtext, ':', vurl)




# =============================================================================
# TEST 4. 선택자 사용하기 | select, select_one 사용
# =============================================================================
# 선택자(요소를 전체적인 느낌으로 정의하는 방식) 기본 서식
# - *           : 모든 요소를 선택
# - <요소명>     : 특정 요소를 선택
# - .<클래스이름> : 특정 클래스를 선택
# - #<id이름>    : 특정 속성 선택


# 선택자들 관계 지정
# - <선택자> , <선택자>   : 여러 선택자 모두 선택
# - <선택자> <선택자>     : 앞 선택자 하위에 있는 다음 선택자 지정
# - <선택자> > <선택자>   : 앞 선택자 하위에 있는 바로 다음 선택자 지정
# - <선택자> + <선택자>   : 같은 계층에서 바로 뒤에 오는 요소 지정
# - <선택자1> ~ <선택자2> : 선택자1에서 선택자2의 모든 요소 지정


# 1. 라이브러리 로딩


# 2. html 가져오기
#    - div       : 특별한 의미는 없으며 문서의 영역을 지정하는데 사용 요소
#    - id 속성    : 식별자
#    - class 속성 : 정의를 내리기 위한 구분자    

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




# =============================================================================
# TEST 5. 위키 문헌의 윤동주 작가의 작품 목록 가져오기
# =============================================================================
# 1. url 가져오기
from urllib.request import urlopen 

url = "https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%EC%9C%A4%EB%8F%99%EC%A3%BC"
html = urlopen(url)


# 2. html parsing 
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')


# 3. 작품 이름 가져오기 위한 CSS 선택자 확인
#    1)우클릭 하여 개발자 도구 확인  
#    2) 원하는 부분을 선택하여 html 요소 확인(우클릭 > 복사 > select)
#       mw-content-text > div.mw-parser-output > ul:nth-child(6) > li > b > a   


# 4. 추출
vlist = soup.select('a')                                                       # 이용 약관까지 출력
vlist = soup.select('div.mw-parser-output > ul > li a')                        # 책 목록만 출력
vlist = soup.select('div#mw-content-text > div.mw-parser-output > ul > li a')  # 책 목록만 출력

output = [] 
for i in vlist : 
    output.append(i.string)
    



# =============================================================================
# TEST 6. 아디다스 운동화 목록 가져오기
# =============================================================================
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

for page_num in range(1,11) :
    url = 'https://sneakernews.com/category/adidas/page/' + str(page_num)
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    vlist = soup.select('div.post-content > h4 > a')
    
    vname = [ a_name.string for a_name in vlist]
    outlist.append(vname)

df_name = DataFrame(outlist, index = range(1,11))
df_name = df_name.stack().reset_index()
df_name = df_name.drop('level_1', axis=1)
df_name.columns = ['page_num', 'name']



# 모든 페이지에 있는 정보 가져오기(페이지 정보가 없는 경우 크롤링 stop)
from urllib.request import HTTPError
from urllib.request import urlopen
from bs4 import BeautifulSoup

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

f_page(1100)
    
def adidas_crw() :
    page_num = 1
    df_result = []
    while True :
        vname = f_page(page_num)
        if vname is None :
            break
        else : 
            df_result.append({'page':page_num, 'name':vname})
        page_num += 1                 # page_num = page_num + 1
    return df_result


import time
start = time.time()
d1 = adidas_crw()
end = time.time()

print(end - start)


    

# =============================================================================
# TEST 7. 네이버 날씨 가져오기(특정 지역)
# =============================================================================
import requests
from bs4 import BeautifulSoup

# 1. url 확인하기
#    다음의 두 형태로 선택 가능
vaddr = '강릉'
vhtml = requests.get('https://search.naver.com/search.naver?query='+ vaddr +'날씨')
vhtml = requests.get('https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query='+ vaddr +'날씨')


# 2. error code 확인
#    200 : 정상
#    404 : Not Found
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
# TEST 8. selenium을 이용한 네이버 검색 
# =============================================================================
# pip install selenium
# pip install webdriver_manager

# 1. 필요 패키지 로딩
from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# 2. Chrome 실행
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.implicitly_wait(30)                     # driver에 대한 time out 설정(초 단위)


# 3. Chrome을 통해 url 접속
url = 'http://www.naver.com'
driver.get(url)


# 4. 검색창에 텍스트 입력하기
#    - 개발자도구를 통해 검색창 입력에 필요한 요소 정보 체크
#    - id = "query" 임을 확인
#    - 버튼을 클릭하는 형식의 모듈 필요
element = driver.find_element(By.ID, 'query')  # find_element : 요소를 찾는 메서드
element.send_keys('피자')
element.send_keys(Keys.ENTER)



# [ 참고 ] 기타 동작
driver.back()                     # 뒤로가기
driver.forward()                  # 앞으로 가기
driver.refresh()                  # 새로고침
driver.close()                    # 탭 닫기
driver.quit()                     # 창 닫기
driver.maximize_window()          # 창 최대화
driver.minimize_window()          # 창 최소화
print(driver.page_source)         # 브라우저 HTML 정보 출력




# [ selenium을 이용한 네이버 로그인 ]
# 1. 필요 패키지 로딩
from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# 2. Chrome 실행
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.implicitly_wait(30)      # driver에 대한 time out 설정(초 단위)


# 3. Chrome을 통해 url 접속
url = 'http://www.naver.com'
driver.get(url)


# 4. 로그인 시도
#    로그인 버튼이 정의된 요소를 개발자도구를 통해 확인
#    현재시점 class명 : link_login
element = driver.find_element(By.CLASS_NAME, 'link_login')
element.click()

driver.find_element(By.ID, 'id').send_keys('')
driver.find_element(By.ID, 'pw').send_keys('')
driver.find_element(By.ID, 'log.login').click()
# bot 인지, 로그인 막힘


# 5. 클립보드를 사용한 로그인 시도
user_id = ''
user_pw = ''


# Chrome 실행, 로그인 버튼 클릭까지 진행
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.implicitly_wait(30) 
url = 'http://www.naver.com'
driver.get(url)
element = driver.find_element(By.CLASS_NAME, 'link_login')
element.click()


# id, pw 복사&붙여넣기
# pip install pyperclip
import pyperclip
import time

ele_id = driver.find_element(By.ID, 'id')
ele_id.click()
pyperclip.copy(user_id)                  # user_id에 있는 내용 클립보드에 복사
ele_id.send_keys(Keys.CONTROL, 'v')      # 클립보드 내용 붙여넣기
time.sleep(1)

ele_pw = driver.find_element(By.ID, 'pw')
ele_pw.click()
pyperclip.copy(user_pw)                  # user_pw에 있는 내용 클립보드에 복사
ele_pw.send_keys(Keys.CONTROL, 'v')      # 클립보드 내용 붙여넣기

driver.find_element(By.ID, 'log.login').click()




# [ 네이버 뉴스 - 헤드라인, 미리보기 가져오기 ]
from urlib.request import urlopen
from bs4 import BeautifulSoup

# 첫페이지 뉴스 정보 가져오기
url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EB%89%B4%EC%8A%A4&sort=0&photo=0&field=0&pd=0&ds=&de=&cluster_rank=12&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:all,a:all&start=1'
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')


# 헤드라인 가져오기
vlist = soup.select('div.news_area > a')
for news in vlist :
    print(news.text)


# 미리보기 가져오기
vlist = soup.select('div.dsc_wrap > a')
for news in vlist :
    print(news.text)



# 특정 페이지까지 정보 가져오기 함수
# 1. url 정의
#    - 페이지 변화가 html에서 어떻게 이루어지는지 우선 파악해야 함
#    - url 마지막 숫자가 변경되는 것을 확인
page = 2
if page == 1 :
    page = ''
else :
    page -= 1
url = f'http://...start={page}1'


# 2. 지정 페이지의 뉴스 크롤링 함수
def f_naver_news_crawl(page:int) :
    # page 변화에 따른 url 생성
    if page == 1 :      # page 1일 경우 페이지번호 생략을 위해 빈문자열 생성
        page = ''
    else :
        page -= 1       # page = page - 1  
    url = f'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EB%89%B4%EC%8A%A4&sort=0&photo=0&field=0&pd=0&ds=&de=&cluster_rank=12&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:all,a:all&start={page}1'
    
    # url open 및 파싱
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    
    # 헤드라인 추출
    vlist = soup.select('div.news_area > a')
    head = [news.text for news in vlist]

    # 미리보기 추출
    vlist2 = soup.select('div.dsc_wrap > a')
    preview = [prev.text for prev in vlist2]
    
    # 최종 리턴 형식(데이터프레임화)
    df_news = DataFrame({'head' : head, 'preview' : preview})
    return df_news

f_naver_news_crawl(5)


# 3. 특정 페이지까지 모든 뉴스 크롤링 함수
def f_naver_news_all_crawl(page:int) :
    df_total = DataFrame()
    for page_num in range(1, page + 1) :
        df_news = f_naver_news_crawl(page)
        df_total = pd.concat([df_total, df_news], ignore_index = True)
        df_news = DataFrame()
    return df_total

f_naver_news_all_crawl(3)




# =============================================================================
# TEST 9. 베스트셀러 도서목록 가져오기
# =============================================================================
# 특이사항 : urlopen, requests, httpx 사용 불가
# 책 제목, 저자, 출판사, 가격 크롤링

# 1. 필요 패키지 로딩
from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


# 2. url 정의 및 호출
url = 'https://book.naver.com/bestsell/bestseller_list.naver?cp=yes24&cate=total&bestWeek=2022-07-4&indexCount=&type=list&page=1'

# 1) urlopen
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

# 2) requests
html = requests.get(url)
soup = BeautifulSoup(html, 'html.parser')

# 3) httpx
html = httpx.get(url)
soup = BeautifulSoup(html, 'html.parser')

# 4) selenium
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.implicitly_wait(30) 
driver.get(url)
soup = BeautifulSoup(driver.page_source, 'html.parser')


# 첫번째 페이지에 대한 정보 크롤링 함수
def bestseller_list():
    url = 'https://book.naver.com/bestsell/bestseller_list.naver?cp=yes24&cate=total&bestWeek=2022-07-4&indexCount=&type=list&page=1'
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(30) 
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    vlist = soup.select('div.section_bestseller li > dl a.N\=a\:bel\.title')
    vtitle = [i.text for i in vlist]
    vlist2 = soup.select('div.section_bestseller li > dl > dd.txt_block')
    vinfo = [j.text.split('|') for j in vlist2]
    vlist3 = soup.select('div.section_bestseller li > dl > dd.txt_desc em.price')
    vprice = [z.text for z in vlist3]
 
  
    df_book = DataFrame({'책제목' : vtitle, 
                         '저자' : Series(vinfo).str[0].str[2:],
                         '출판사' : Series(vinfo).str[1].str[2:],
                         '가격' : Series(vprice).str[:-7].str.replace(',', '')})
    return df_book
    
bestseller_list()


# 지정 페이지까지 모든 정보 크롤링 함수




