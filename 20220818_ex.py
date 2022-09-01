# -*- coding: utf-8 -*-


# 1. 회문 판별 함수 생성
# 회문 : 앞으로 읽어도 뒤로 읽어도 같은 글자
def f_scan(x) :
    i = 0
    vresult = []
    while len(x) > i :
        if x[i] == x[-(i + 1)] :
            vresult.append(1) 
        else :
            vresult.append(-1)
        i = i +1
    if -1 in vresult :
        print('회문이 아닙니다.')
    else :
        print('회문입니다.')
        
f_scan('abcdedcba')
f_scan('accbbcca')
f_scan('abcde')




# 2. f_instr 함수 생성
# f_instr(data,str,start,n) => data에서 str을 start 위치에서 n번째 발견된 위치 리턴
def f_instr(data, vstr, start = 0, n = 1) :
    if data[start:].count(vstr) < n :
        return -1 
    else : 
        for i in range(0, n) :
            pos = data.find(vstr, start)
            start = pos + len(vstr)
    return pos
    
f_instr('a@b@c@d@', '@')
f_instr('a@b@c@d@', '@', 2, 2)




# 3. professor.csv 파일을 array 형식으로 불러온 뒤 다음 수행(컬럼명은 제외)
import pandas as pd
pro = pd.read_csv('data/professor.csv', encoding='cp949')

# 1) email_id 출력
# map
f_split = lambda x : x.split('@')[0]
list(map(f_split, pro['EMAIL']))

# for
vid = []
for i in pro['EMAIL'] :
    vid.append(i.split('@')[0])



# 2) 입사연도가 1990년 이전인 교수는 PAY의 15% 인상, 
#    90년 포함 이후인 경우는 10% 인상하여 출력
# for
new_sal = []
for i, j in zip(pro['HIREDATE'], pro['PAY']) :
    if int(i[:4]) < 1990 :
        new_sal.append(round(j * 1.15))
    else :
        new_sal.append(round(j * 1.1))
        

# map
def f_newsal(x, y) :
    if int(x[:4]) < 1990 :
        vnew_sal = round(y * 1.15)
    else :
        vnew_sal = round(y * 1.1)
    return vnew_sal
        
list(map(f_newsal, pro['HIREDATE'], pro['PAY']))



