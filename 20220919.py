# -*- coding: utf-8 -*-

# =============================================================================
# 나이브 베이즈 모델
# =============================================================================
# text data의 분류, 문자 분류, 스팸 여부 등 확인 사용(텍스트 데이터 분석)
# 조건부 확률(ex. 대출이라는 단어가 포함되어 있을 때 스팸, 햄의 확률 계산)을 사용하여 분류
# 계산이 간단, 빠름



#[ 연습 문제 ]
# 희귀병 검사 > 병에 걸린 사람을 D, 양성 판정을 받은 사람을 P
# 병에 걸렸을 경우 양성 판정이 나올 확률 P(P|D) : 0.95
#                                   P(P|Dᶜ) : 0.05
# 전체 인구 중 실제 병에 걸릴 확률 P(D) : 0.001

# P(P) = ?
# P(D) = 0.001
# P(Dᶜ) = 1 - 0.001 = 0.999

# 양성 판정이 나왔을 때(P), 실제 병에 걸렸을 확률(D) 
# P(D|P) = P(P|D)*P(D) / P(P)
#        = P(P|D)*P(D) / (P(P|D)*P(D) + P(P|Dᶜ)*P(Dᶜ))




# [ 예제 ] 나이브 베이즈 모델 스팸분류 자연어 처리 과정
# 1. 데이터 만들기
vlist = ['안녕하세요 은혜 금융입니다. 귀하는 아무 조건 없이 1000만원 대출 가능한 우수 고객입니다.' ,
'새로운 상품이 출시되어 안내문자 드립니다. 21세 이상 직장인/사업자 대상 무보증 최고 1억원 대출 가능합니다.',
'안녕하세요. 잘 지내시죠? 안부차 연락드립니다. 항상 건강하세요.',
'오늘의 집에서 고객님께만 20% 할인 쿠폰을 드렸어요. 지금 바로 쿠폰함을 확인해보세요.',
'고객님 카드론 혜택 안내드립니다. 필요할 때 딱 이자는 쓴 만큼만 딱 장기카드 대출 가능합니다.']

y = [1, 1, 0, 0, 1]                      # 스팸문자 : 1, 햄 : 0


# 2. 토큰화 및 조사제거
import konlpy.tag

Okt = konlpy.tag.Okt()                   # 자연어 사전 정의
Okt.pos(vlist[0])                        # 문장 분석


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
#       CountVectorizer : 문장(문자열)에서 단어 추출(토큰화 > 불용어 제거), 단어별로 빈도수 계산
from sklearn.feature_extraction.text import CountVectorizer    
    
vect1 = CountVectorizer(analyzer='word', stop_words=stops)
X = vect1.fit_transform(outlist)

vwords = vect1.get_feature_names()       # get_feature_names > get_feature_names_out 변경 예정
X.toarray()                              # 각 문장별 토큰화된 단어의 수 배열 형태로 제공

DataFrame(X.toarray(), columns = vwords)


# 6. TF-IDF 변환
#    단어가 많이 반복되는 경우 > 해당 단어 중요도 높게 계산
#    모든 문서(관측치)마다 반복되는 단어의 경우 가중치를 다시 낮게 반영 필요

#    1) TF(Term Frequency) : 단어의 빈도수. 각 문서에 각 단어가 포함된 횟수
#    2) IDF(Inverse Document Frequency) : Document Frequency의 역수
#    3) DF : 특정 단어가 포함된 문장 수 / 전체 문장 수 > 공통적으로 발견되는 문자인지 확인
#    4) IDF = log(전체 문장 수/(특정 단어가 포함된 문장 수 + 1))

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
m_tfidf = TfidfTransformer()
X_tfidf = m_tfidf.fit_transform(X)       # X : 단어별 count 결과를 가져야함

print(X_tfidf)                           # (0, 14)	       0.5065627726821486
                                         # 문장번호, 단어번호, ifidf 변환값
                
                
# 7. 나이브 베이즈 모델 적용
#    - GaussianNB    : 설명변수가 연속형인 경우
#    - CategoricalNB : 설명변수가 범주형인 경우
#    - BernoulliNB   : 설명변수가 범주형인 경우(종속변수가 이진분류)
#    - MultinomialNB : 단어수(문장에 대한 분리) 기반 모델

from sklearn.naive_bayes import MultinomialNB

# 1) TF-IDF 변환 이전
m_nb1 = MultinomialNB()
m_nb1.fit(X, y)
m_nb1.score(X, y)                        # 100%

m_nb1.predict_proba(X)[:,1]              # 스팸일 확률

# 2) TF-IDF 변환 이후
m_nb2 = MultinomialNB()
m_nb2.fit(X_tfidf, y)
m_nb2.score(X_tfidf, y)                  # 100%

m_nb2.predict_proba(X_tfidf)[:,1]        # 스팸일 확률


# 8. 워드 클라우드 생성
#    pip install wordcloud(visual studio C++ 설치 필요)
from wordcloud import WordCloud

from os import path
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'# 맑은고딕체 폰트설정

noun_text = ' '.join(vwords)             # 표현할 단어들이 포함된 문자열 생성

wordcloud = WordCloud(max_font_size=50, max_words=30, background_color='white', 
                      relative_scaling=.5, font_path=FONT_PATH).generate(noun_text)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


