#!/usr/bin/env python3
# -*- coding: utf-8 -*-
run my_profile

# =============================================================================
# 파일 불러오기
# =============================================================================
# 1. 기본 제공(open-fetch-close 형식 제공)
#    - open : 내부 메모리 영역(커서) 생성 후 외부 파일 결과를 메모리에 저장
#    - fetch : 메모리 영역에 저장된 데이터를 꺼내는 작업
#    - close : 메모리 영역 해제


# [ 외부 파일 내부 객체로 불러오기 ]
c1 = open('data/file1.txt')
v1 = c1.readline()
v2 = c1.readline()
c1.close()


# - 모든 라인 가져오기(for)
c1 = open('data/file1.txt')
while 1: 
    v1 = c1.readline()
    if v1 == '' :
        break
    outlist.append(v1)
c1.close()


# - 모든 라인 가져오기(readlines)
c1 = open('data/file1.txt')
c1.readlines()


# - 객체를 외부 파일에 쓰기
l2 = [[1, 2, 3], [4, 5, 6]]
c1 = open('write_text.txt', 'w')
c1.writelines(str(l2))                       # 리스트를 강제로 문자열로 변환 후 저장
c1.close()                                   # (리스트 형식 모양 그대로 저장됨)


# - 한줄씩 쓰기
def f_write_txt(file, list, sep) :
    c1= open(file, 'w')
    for i in list :
        vstr = ''
        for j in i :
            vstr = vstr + str(j) + sep
        vstr = vstr.rstrip(sep)
        c1.writelines(vstr + '\n')
    c1.close()
    
f_write_txt('write_text.txt', l2, sep = ',')


# 2. numpy
#    - np.loadtxt
#    - np.savetxt


# 3. pandas
#    - pd.read_csv, pd.read_table, pd.read_excel
#    - df.to_csv, df.to_excel




