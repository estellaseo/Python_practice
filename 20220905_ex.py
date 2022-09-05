# -*- coding: utf-8 -*-
run my_profile

# 1. shoppingmall.txt 파일을 읽고
df1 = pd.read_csv('data/shoppingmall.txt', encoding = 'cp949', sep = '|', header = None)

# 쇼핑몰 이름과 쇼핑몰 주소 정보만 담아 데이터프레임 생성
# ========================================================================
# 22. 

# 예쁨주의보 시크릿라벨              => 쇼핑몰 이름(전체 가져가기)
# 예쁨주의보 시크릿라벨 네이버페이 
   
# 센스있는 여자의 선택 시크릿라벨, 시선강탈 마성매력의 트렌드룩으로 여성미를 살리다

# http://www.secretlabel.co.kr   => 쇼핑몰 주소

# 광고집행기간  61개월 이상 
# ========================================================================

vhtml = df1.iloc[:, 0].str.findall('http://.+').str[0].dropna().str.strip().reset_index(drop = True)

vidx = df1.iloc[:, 0].str.findall('[0-9]+\. ').str[0].dropna().index
vname = df1.iloc[vidx+1, 0].reset_index(drop = True)

DataFrame({'NAME':vname, 'HPAGE':vhtml})



# 2. oracle_alert_testdb.log 파일을 읽고
df2 = pd.read_csv('data/oracle_alert_testdb.log', sep = '|', header = None)

# 에러코드와 에러내용을 아래 데이터프레임 형식으로 저장

# code             error
# 1109    signalled during: ALTER DATABASE CLOSE NORMAL...

'ORA-1109 signalled during: ALTER DATABASE CLOSE NORMAL...'

import re
r1 = re.compile('(ORA\-[0-9]+)([signalled :.+] .+)')
vre = df2.iloc[:, 0].str.findall(r1).str[0].dropna()

vcode = vre.str[0].str.strip()
verror = vre.str[1].str[2:].str.strip()
DataFrame({'CODE':vcode, 'ERROR':verror})
