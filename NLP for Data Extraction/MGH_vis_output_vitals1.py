import copy
import numpy as np
from unidecode import unidecode
import csv
import re
import os
csv.field_size_limit(1000000000)


out = open('vis_out_vitals.csv', 'a', newline='', encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')

files = sorted(os.listdir('visit/'))
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
                'visit/' +
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

    for (i, line) in enumerate(lines):
        if started == 0:
            if line[0].find(
                    'Vitals') != -1 or line[0].find('Enc _s Group') != -1 or line[0].find('Exam:') != -1:
                started = 1
                n += 1

        if started == 1:
            if assert_stop(line, stop_list_disease) == 1:
                started = 0
                continue
            '''if line[0][0]=='?':
				disease=line[0].split('    ')[0][1:].strip()
				print(disease)
				D.append(disease)'''

            BP_ind = [
                i.span()[0] for i in re.finditer(
                    '(BP|blood pressure|Blood pressure)',
                    line[0])]
            if BP_ind != [] and BP == '':
                text = line[0][BP_ind[0]:]
                text = remove_time(text)
                text = remove_date(text)
                # date_list=re.findall('(\d{1,4}\/\d{1,4}\/\d{1,4})',line[0])
                r = re.findall(r'(\d{2,3}\/\d{2,3})', text)
                # if r!=[]:
                for bp in r:
                    if BP == '':
                        BP = bp
                        break

            #Temp_ind=[i.span()[0] for i in re.finditer('(Temp|Temperature|temperature)', line[0])]
            # if Temp_ind!=[] and Temp=='':
            if Temp == '':
                # text=line[0][Temp_ind[0]:]
                text = line[0]
                text = remove_time(text)
                text = remove_date(text)
                # date_list=re.findall('(\d{1,4}\/\d{1,4}\/\d{1,4})',line[0])
                r = re.findall(r'([\d{2}\.]+ \?C)', text)
                # if r!=[]:
                for tp in r:
                    if Temp == '':
                        Temp = tp
                        break

            Pul_ind = [
                i.span()[0] for i in re.finditer(
                    r'(Pulse|pulse|Heart Rate|\ P\ |\ P\:|HR)',
                    line[0])]
            if Pul_ind != [] and Pul == '':
                text = line[0][Pul_ind[0]:]
                text = remove_time(text)
                text = remove_date(text)
                date_list = re.findall(r'(\d{1,4}\/\d{1,4}\/\d{1,4})', line[0])
                r = re.findall(
                    r'((?<!SpO)\d{2,3}(?! \?C| \?F|\.\d \?C|\.\d \?F| \%|%))', text)
                # if r!=[]:
                for p in r:
                    if Pul == '':
                        Pul = p
                        break

            RR_ind = [
                i.span()[0] for i in re.finditer(
                    r'(Respiratory Rate|RR\ |Resp|RR:|resp. rate|Respiratory)',
                    line[0])]
            if RR_ind != [] and RR == '' and line[0].find('RRR') == -1:
                if line[0][RR_ind[0]:RR_ind[0] + 5].strip() == 'RR':
                    RR == ''
                else:
                    # print(filename)
                    text = line[0][RR_ind[0]:]
                    # print(text)
                    text = remove_time(text)
                    text = remove_date(text)

                    # print(text)
                    date_list = re.findall(
                        r'(\d{1,4}\/\d{1,4}\/\d{1,4})', line[0])
                    r = re.findall(
                        r'((?<!SpO)\d{1,3}(?! \?C| \?F|\.\d \?C|\.\d \?F| \%|%))', text)
                    for rr in r:
                        if RR == '':
                            RR = rr
                            break

            SP_ind = [i.span()[0] for i in re.finditer('(SpO2)', line[0])]
            if SP_ind != [] and SP == '':
                # text=line[0][Temp_ind[0]:]
                text = line[0]
                text = remove_time(text)
                text = remove_date(text)
                # date_list=re.findall('(\d{1,4}\/\d{1,4}\/\d{1,4})',line[0])
                r = re.findall(r'(\d{2,3}\ ?\%)', text)
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
    out_line = ['Vitals in visit(1', filename, Time, BP, Temp, Pul, RR, SP]
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
