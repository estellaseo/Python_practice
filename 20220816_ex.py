# -*- coding: utf-8 -*-

# 1. 지폐 계산 프로그램
# 사용자한테 금액을 입력받고 
# 50000원권, 10000원권, 5000원권, 1000원권, 나머지 금액 출력

# 10000  :   2장
#  5000  :   1장
#  1000  :   6장
# 나머지  :   2원

import math

total = input('금액을 입력하세요. ')
total = int(total)
c50 = math.trunc(total/50000)
c10 = math.trunc((total - c50 * 50000)/10000)
c5 = math.trunc((total - c50 * 50000 - c10 * 10000)/5000)
c1 = math.trunc((total - c50 * 50000 - c10 * 10000 - c5 * 5000)/1000)
c0 = total - c50 * 50000 - c10 * 10000 - c5 * 5000 - c1 * 1000
print('오만 원 : ', c50, '장')
print('만 원 : ', c10, '장')
print('오천 원 : ', c5, '장')
print('천 원 : ', c1, '장')
print('나머지 : ', c0, '원')



# 2. 이메일 주소를 입력받고 다음과 같이 출력

# 아이디 : a1234
# 메일엔진 : naver    
# 홈페이지 : http://itwill.com/a1234

vemail = input('이메일 주소를 입력하세요 : ')
vid = vemail.split('@')
vengine = vid[1].split('.')[0]
print('아이디 : ', vid[0])
print('메일엔진 : ', vengine)
print('홈페이지 : ', 'http://itwill.com/'+vid[0])



# 3. 문자열, 찾을 문자열, 바꿀 문자열을 입력 받아 변경한 결과를 아래와 같이 출력
# 전 :
# 후 :
    
s1 = input('문자열을 입력하세요 : ')
s2 = input('찾을 문자열을 입력하세요 : ')
s3 = input('바꿀 문자열을 입력하세요 : ')

print('전 : ', s1)
print('후 : ', s1.replace(s2, s3))

    

# 4. num1='12,000' 의 값을 생성 후, 33으로 나눈 값을 소숫점 둘째짜리까지 표현    
num1 = '12,000'
num2 = int(num1.replace(',', ''))
round(num2/33, 2)







