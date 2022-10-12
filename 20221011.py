# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
#  자연어 처리
# =============================================================================
# - 텍스트 처리
#   1) 토큰화(문장을 단어화)
#   2) 일반적인 단어로 변경(자연어 사전에 등록된 단어 추출, 조사 제거)    
#   3) 불용어 정의 삭제


# [ 한국어 자연어 처리 환경 설정 ]
# 1. JDK 설치 및 path 등록
# www.java.com 다운로드 및 설치
# 설치 확인 : C:Program Files/Java 폴더 생성 여부
# 환경설정에서 path 등록 => C:\Program Files\Java\jre1.8.0_341\bin

# 2. JPype 다운로드 및 설치
# 주의 : 파이썬 버전(Python 3.9.12), 윈도우(32/64bit) 버전 확인 후 설치
# 파이썬 버전 확인 : cmd > python --version

# 1) 프로그램 다운 : https://www.lfd.uci.edu/~gohlke/pythonlibs/ 
#                   => JPype1‑1.4.0‑cp39‑cp39‑win_amd64.whl 다운                
# 2) 위 파일을 cmd 홈디렉토리로 이동
# 3) cmd에서 pip install JPype1‑1.4.0‑cp39‑cp39‑win_amd64.whl
#    (파일명 주의 : window 상에서 표현되는 이름이 다를 수 있음 => dir로 파일명 확인)

# 3. konlpy : 한국어 자연어 처리시 사용
# pip install konlpy(cmd에서 실행)
# spyder restart

import konlpy




# =============================================================================
# 나이브 베이즈 모델(스팸 분류)
# =============================================================================
# [ 예제 - 스팸분류 자연어 처리 과정 ]
# 1. 데이터 만들기
vlist = ['안녕하세요 은혜 금융입니다. 귀하는 아무 조건 없이 1000만원 대출 가능한 우수 고객입니다.' ,
'새로운 상품이 출시되어 안내문자 드립니다. 21세 이상 직장인/사업자 대상 무보증 최고 1억원 대출 가능합니다.',
'안녕하세요. 잘 지내시죠? 안부차 연락드립니다. 항상 건강하세요.',
'오늘의 집에서 고객님께만 20% 할인 쿠폰을 드렸어요. 지금 바로 쿠폰함을 확인해보세요.',
'고객님 카드론 혜택 안내드립니다. 필요할 때 딱 이자는 쓴 만큼만 딱 장기카드 대출 가능합니다.']

y = [1, 1, 0, 0, 1]     # 스팸문자 : 1, 햄 : 0


# 2. 토큰화 및 조사제거
import konlpy.tag

Okt = konlpy.tag.Okt()     # 자연어 사전 정의
 
Okt.pos(vlist[0])          # 문장 분석


# 3. 명사 추출
def f_extract(text) :
    Otk = konlpy.tag.Okt()
    Otk_v1 = Otk.pos(text) 
    
    Noun_words = []
    for word, pos in Otk_v1 :
        if pos == 'Noun' :               # 명사외에 다른 대상을 추출할 경우 수정 필요
            Noun_words.append(word)
            
    return Noun_words

f_extract(vlist[0])                      # 명사만 추출된 리스트


# 4. 전체 데이터 셋 적용(각 문장에서 추출된 명사는 다시 한 문장으로 결합)
outlist = []
for i in vlist :
    outlist.append(' '.join(f_extract(i)))


# 5. 불용어 처리(제거)
#    1) 불용어 정의
stops = ['은혜', '아무', '차', '만', '때', '이상', '세', '무', '항상', '오늘', '집', '바로']

#    2) 불용어 제거 및 단어의 수 count
#       CountVectorizer : 문장(문자열)에서 단어 추출(토큰화 -> 불용어 제거), 단어별로 빈도수 계산
from sklearn.feature_extraction.text import CountVectorizer    
    
vect1 = CountVectorizer(analyzer='word', stop_words=stops)
X = vect1.fit_transform(outlist)

vwords = vect1.get_feature_names()       # get_feature_names => get_feature_names_out 변경 예정
X.toarray()                              # 각 문장별 토큰화된 단어의 수 배열 형태로 제공

DataFrame(X.toarray(), columns = vwords)


# 6. TF-IDF 변환
#    단어가 많이 반복되는 경우 > 해당 단어 중요도 높게 계산
#    모든 문서(관측치)마다 반복되는 단어의 경우 가중치를 다시 낮게 반영 필요

#    1) TF(Term Frequency) : 단어의 빈도수. 각 문서에 각 단어가 포함된 횟수
#    2) IDF(Inverse Document Frequency) : Document Frequency의 역수
#    3) DF : 특정 단어가 포함된 문장 수 / 전체 문장 수   => 공통적으로 발견되는 문자인지를 확인!
#    4) IDF = log(전체 문장 수/(특정 단어가 포함된 문장 수 + 1))

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
m_tfidf = TfidfTransformer()
X_tfidf = m_tfidf.fit_transform(X)         # X : 단어별 count 결과를 가져야함

print(X_tfidf)                         # (0, 14)	           0.5065627726821486
                                       # 문장번호, 단어번호,  ifidf 변환값
                
                
# 7. 나이브 베이즈 모델 적용
#    - GaussianNB : 설명변수가 연속형인 경우
#    - CategoricalNB : 설명변수가 범주형인 경우
#    - BernoulliNB : 설명변수가 범주형인 경우(종속변수가 이진분류)
#    - MultinomialNB : 단어수(문장에 대한 분리) 기반 모델

from sklearn.naive_bayes import MultinomialNB

# 1) TF-IDF 변환 이전
m_nb1 = MultinomialNB()
m_nb1.fit(X, y)
m_nb1.score(X, y)              # 100%

m_nb1.predict_proba(X)[:,1]    # 스팸일 확률

# 2) TF-IDF 변환 이후
m_nb2 = MultinomialNB()
m_nb2.fit(X_tfidf, y)
m_nb2.score(X_tfidf, y)              # 100%

m_nb2.predict_proba(X_tfidf)[:,1]    # 스팸일 확률


# 8. 워드 클라우드 생성
# pip install wordcloud(visual studio C++ 설치 필요)
from wordcloud import WordCloud

from os import path
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'    # 맑은고딕체 폰트설정

noun_text = ' '.join(vwords)                 # 표현할 단어들이 포함된 문자열 생성

wordcloud = WordCloud(max_font_size=50, max_words=30, background_color='white', 
                      relative_scaling=.5, font_path=FONT_PATH).generate(noun_text)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




# =============================================================================
# 참고 - wordcloud 설치시 Visual C++ 설치 필요 에러 해결
# =============================================================================
# 1. Visual C++ Build Tools 다운
# https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/

# => C++ Build Tools 다운로드 클릭 후 설치 진행

# 2. 설치 중 WorkLoads 선택 부분에서 Desktop development with C++ 클릭 후 설치 진행
# 3. 컴퓨터 재시작
# 4. cmd > pip install wordcloud



# [ 연습 문제 ] 뉴스 데이터(영어) 문서 분류
# 1. 데이터 불러오기
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups()

news_x = news['data']
news_y = news['target']

news['target_names']        # 20개의 기사 분류
type(news_x)
len(news_x)                 # 11314개의 뉴스 데이터
 
print(news_x[0])            # 첫번째 뉴스 데이터 출력


# 2. 필요 패키지 불러오기
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 3. 데이터 변환
# 1) 토큰화 및 단어수 세기 : CountVectorizer 사용
m_count = CountVectorizer()
news_x_count = m_count.fit_transform(news_x)

# 2) TF-IDF 변환(불필요하게 전체적으로 반복되는 단어에 대해 가중치 줄이는 기법) : TfidfTransformer 사용
m_tfidf = TfidfTransformer()
news_x_count_tfidf = m_tfidf.fit_transform(news_x_count)

# 4. 데이터 분리(train/test)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(news_x_count, news_y, random_state=0)
train_x_tf, test_x_tf, train_y_tf, test_y_tf = train_test_split(news_x_count_tfidf, news_y, random_state=0)
    
# 5. 나이브 베이즈 모델 적용
# 1) TF-IDF 변환 이전
m_nb1 = MultinomialNB()
m_nb1.fit(train_x, train_y)
m_nb1.score(train_x, train_y)    # 92.52
m_nb1.score(test_x, test_y)      # 82.43
 
# 2) TF-IDF 변환 이후
m_nb2 = MultinomialNB()
m_nb2.fit(train_x_tf, train_y_tf)
m_nb2.score(train_x_tf, train_y_tf)    # 94.00
m_nb2.score(test_x_tf, test_y_tf)      # 84.62

m_nb2.predict_proba(test_x_tf)
m_nb2.predict_proba(test_x_tf).argmax(axis=1)     # 각 문서마다 최대확률을 갖는 Y값(위치) 확인
m_nb2.predict(test_x_tf)                          # 각 문서마다 예상되는 문서 분류 값(Y값)

# 6. 예측
vpre = m_nb2.predict(test_x_tf[0])[0]     # predict value([0] : scalar return을 위한 색인)
news['target_names'][vpre]



