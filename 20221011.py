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
#                  > JPype1‑1.4.0‑cp39‑cp39‑win_amd64.whl 다운                
# 2) 위 파일을 cmd 홈디렉토리로 이동
# 3) cmd에서 pip install JPype1‑1.4.0‑cp39‑cp39‑win_amd64.whl
#    (파일명 주의 : window 상에서 표현되는 이름이 다를 수 있음 => dir로 파일명 확인)

# 3. konlpy : 한국어 자연어 처리시 사용
# pip install konlpy(cmd에서 실행)
# spyder restart

import konlpy




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
#    1) 토큰화 및 단어수 세기 : CountVectorizer 사용
m_count = CountVectorizer()
news_x_count = m_count.fit_transform(news_x)

#    2) TF-IDF 변환(불필요하게 전체적으로 반복되는 단어에 대해 가중치 줄이는 기법) : TfidfTransformer 사용
m_tfidf = TfidfTransformer()
news_x_count_tfidf = m_tfidf.fit_transform(news_x_count)


# 4. 데이터 분리(train/test)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(news_x_count, news_y, random_state=0)
train_x_tf, test_x_tf, train_y_tf, test_y_tf = train_test_split(news_x_count_tfidf, news_y, random_state=0)
    

# 5. 나이브 베이즈 모델 적용
#    1) TF-IDF 변환 이전
m_nb1 = MultinomialNB()
m_nb1.fit(train_x, train_y)
m_nb1.score(train_x, train_y)                 # 92.52
m_nb1.score(test_x, test_y)                   # 82.43
 
#    2) TF-IDF 변환 이후
m_nb2 = MultinomialNB()
m_nb2.fit(train_x_tf, train_y_tf)
m_nb2.score(train_x_tf, train_y_tf)           # 94.00
m_nb2.score(test_x_tf, test_y_tf)             # 84.62

m_nb2.predict_proba(test_x_tf)
m_nb2.predict_proba(test_x_tf).argmax(axis=1) # 각 문서마다 최대확률을 갖는 Y값(위치) 확인
m_nb2.predict(test_x_tf)                      # 각 문서마다 예상되는 문서 분류 값(Y값)

# 6. 예측
vpre = m_nb2.predict(test_x_tf[0])[0]         # predict value([0] : scalar return을 위한 색인)
news['target_names'][vpre]



