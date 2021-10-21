import copy
import numpy as np
from unidecode import unidecode
import csv
import re
import os
csv.field_size_limit(1000000000)


out = open('hnp_out_vitals2.csv', 'a', newline='', encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')

files = sorted(os.listdir('hnp/'))
# print(files)

n = 0

#stop_list_disease=['medications prior to admission','prior to admission medication','home medications','allergies','family history','social history','no past medical history on file','medications','surgical history','PHYSICAL EXAMINATION'.lower(),'Exam','ROS']
stop_list_disease = [
    'medications prior to admission',
    'prior to admission medication',
    'home medications',
    'allergies',
    'family history',
    'social history',
    'medications',
    'Past Medical History',
    'PAST MEDICAL HISTORY',
    'Patient Active Problem',
    'Past Medical and Surgical History',
    'Plan',
    'PLAN']

started = 0


def findtime(head):
    ls = head.split('|')
    time = 'N/A'
    for t in ls:
        if t.find('AM') != -1 or t.find('PM') != -1:
            time = t
            break

    return time


def assert_stop(line, stop_list):
    a = 0
    for k in stop_list:
        if line[0].lower().find(k) != -1 or line[0].find(k) != -1:
            a = 1
            break
    return a


def IsDate(bp, date_list):
    f = 0
    for d in date_list:
        if d.find(bp) != -1:
            f = 1
            print(bp)
            print(date_list)
            break
    return f


def remove_time(text):
    date_list = [i.span() for i in re.finditer(
        r'(\d{1,4}\/\d{1,4}\/\d{1,4}\s+\d{4})', text)]
    # print(date_list)
    text_r = copy.deepcopy(text)
    for tu in date_list:
        date = text[tu[0]:tu[1]]
        # print(date)
        text_r = text_r.replace(date, '')
    return text_r


def detect_time(text):
    date_list = [i.span() for i in re.finditer(
        r'(\d{1,4}\/\d{1,4}\/\d{1,4}\s+\d{4})', text)]
    # print(date_list)
    '''text_r=copy.deepcopy(text)
	for tu in date_list:
		date=text[tu[0]:tu[1]]
		#print(date)
		text_r=text_r.replace(date,'')'''
    if date_list == []:
        rr = 0
    else:
        rr = 1

    return rr


def remove_date(text):
    date_list = [i.span() for i in re.finditer(
        r'(\d{1,4}\/\d{1,4}\/\d{1,4})', text)]
    # print(date_list)
    text_r = copy.deepcopy(text)
    for tu in date_list:
        date = text[tu[0]:tu[1]]
        # print(date)
        text_r = text_r.replace(date, '')
    return text_r


for filename in files:
    D = []
    # print(filename)
    started = 0
    lines = [
        item for item in csv.reader(
            open(
                'hnp/' +
                filename,
                "r",
                encoding='utf-8'))]

    head = lines[0][0]
    # print(head.split('|'))
    PID = head.split('|')[0]
    # print(PID)
    # Time=head.split('|')[5]
    Time = findtime(head)
    # print(Time)

    BP = ''
    Temp = ''
    Pul = ''
    RR = ''
    SP = ''

    dic = {}
    a = ''

    for (i, line) in enumerate(lines):
        if started == 0:
            if line[0].find('Patient Vitals for the past 24 hrs:') != -1:
                started = 1
                n += 1

        if started == 1:
            if assert_stop(line, stop_list_disease) == 1:
                break
            '''if line[0][0]=='?':
				disease=line[0].split('    ')[0][1:].strip()
				print(disease)
				D.append(disease)'''

            if line[0].find('BP') != -1 and line[0].find('Temp') != - \
                    1 and line[0].find('SpO2') != -1 and dic == {}:
                a = line[0]

                #a='   BP             Temp            Temp   SpO2             Height                Weight'
                T = a.strip().split('  ')
                T = [i.strip() for i in T if i != '']
                # dic={}

                for i in range(len(T)):
                    if T[i] == 'BP' and 'BP' not in dic:
                        dic[T[i]] = i
                    if T[i] == 'Temp' and 'Temp' not in dic:
                        dic[T[i]] = i
                    if T[i] == 'Pulse' and 'Pulse' not in dic:
                        dic[T[i]] = i
                    if T[i] == 'Resp' and 'Resp' not in dic:
                        dic[T[i]] = i
                    if T[i] == 'SpO2' and 'SpO2' not in dic:
                        dic[T[i]] = i

            if detect_time(line[0]) == 1 and dic != {}:
                # print(line[0])
                text = remove_time(line[0])

                # print(text)
                t = text.strip().split('  ')
                t = [i.strip() for i in t if i != '']
                # print(t)

                if len(t) != len(T):
                    continue
                pot_bp = ''
                pot_tp = ''
                pot_p = ''
                pot_rr = ''
                pot_sp = ''

                if 'BP' in dic:
                    pot_bp = t[dic['BP']]
                if 'Temp' in dic:
                    pot_tp = t[dic['Temp']]
                if 'Pulse' in dic:
                    pot_p = t[dic['Pulse']]
                if 'Resp' in dic:
                    pot_rr = t[dic['Resp']]
                if 'SpO2' in dic:
                    pot_sp = t[dic['SpO2']]

                r = re.findall(r'(\d{2,3}\/\d{2,3})', pot_bp)
                # if r!=[]:
                for bp in r:
                    if BP == '':
                        BP = bp
                        break

                # date_list=re.findall('(\d{1,4}\/\d{1,4}\/\d{1,4})',line[0])
                rC = re.findall(r'([\d{2}\.]+ \?C)', pot_tp)
                rF = re.findall(r'([\d{2,3}\.]+ \?F)', pot_tp)

                r = rC
                if r == []:
                    r = rF
                # if r!=[]:
                for tp in r:
                    if Temp == '':
                        Temp = tp
                        break

                r = re.findall(
                    r'((?<!SpO)\d{2,3}(?! \?C| \?F|\.\d \?C|\.\d \?F| \%|%))', pot_p)
                # if r!=[]:
                for p in r:
                    if Pul == '':
                        Pul = p
                        break

                    # print(text)
                    # date_list=re.findall('(\d{1,4}\/\d{1,4}\/\d{1,4})',line[0])
                r = re.findall(
                    r'((?<!SpO)\d{1,3}(?! \?C| \?F|\.\d \?C|\.\d \?F| \%|%))', pot_rr)
                for rr in r:
                    if RR == '':
                        RR = rr
                        break

                # date_list=re.findall('(\d{1,4}\/\d{1,4}\/\d{1,4})',line[0])
                r = re.findall(r'(\d{2,3}\ ?\%)', pot_sp)
                # if r!=[]:
                for sp in r:

                    if SP == '':
                        # print(sp)
                        SP = sp
                        break

            '''pos=[i.span()[0] for i in re.finditer('\?', line[0])]
			#print(pos)

			if pos!=[] and line[0].find('ADM')==-1:
				for i in range(len(pos)):
					if i<len(pos)-1:
						disease=line[0][pos[i]:pos[i+1]].split('  ')[0][1:]
						if disease!='':
							D.append(disease)
							print(disease)
					else:
						disease=line[0][pos[i]:].split('  ')[0][1:]
						if disease!='':
							D.append(disease)
							print(disease)'''

            # if lines[i+1][0].lower().find('diagnosis')!=-1 or lines[i+2][0].lower().find('diagnosis')!=-1:
            # n+=1
            # flag=1
            # break
        # else:
    out_line = ['Vitals in hnp(2', filename, Time, BP, Temp, Pul, RR, SP]
    # out_line.extend(D)
    # ------------- into separate file
    out = open('profile/' + PID + '.csv', 'a', newline='', encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(out_line)
    # ------------- into separate file

    # csv_write.writerow(out_line)

    # if started==0:
    # print(filename)

print(n)
# csv_write.writerow([PID,Time])
