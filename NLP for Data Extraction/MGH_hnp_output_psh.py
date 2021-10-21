
import numpy as np
from unidecode import unidecode
import csv
import re
import os
csv.field_size_limit(1000000000)


out = open('hnp_out_surg.csv', 'a', newline='', encoding='utf-8')
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
    'PHYSICAL EXAMINATION'.lower(),
    'Exam',
    'ROS',
    'EXAM',
    'Vitals']

started = 0


def assert_stop(line, stop_list):
    a = 0
    for k in stop_list:
        if line[0].lower().find(k) != -1 or line[0].find(k) != -1:
            a = 1
            break
    return a


for filename in files:
    D = []
    print(filename)
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
    Time = head.split('|')[5]
    # print(Time)
    for (i, line) in enumerate(lines):
        if started == 0:
            if line[0].find('Past Surgical History') != -1 or line[0].find(
                    'PAST SURGICAL HISTORY') != -1 or line[0].find('Past Medical and Surgical History') != -1:
                started = 1
                n += 1

        if started == 1:
            if assert_stop(line, stop_list_disease) == 1:
                break
            '''if line[0][0]=='?':
				disease=line[0].split('    ')[0][1:].strip()
				print(disease)
				D.append(disease)'''

            pos = [i.span()[0] for i in re.finditer(r'\?', line[0])]
            # print(pos)

            if pos != [] and line[0].find('ADM') == -1:
                for i in range(len(pos)):
                    if i < len(pos) - 1:
                        disease = line[0][pos[i]:pos[i + 1]].split('  ')[0][1:]
                        if disease != '':
                            D.append(disease)
                            print(disease)
                    else:
                        disease = line[0][pos[i]:].split('  ')[0][1:]
                        if disease != '':
                            D.append(disease)
                            print(disease)

            # if lines[i+1][0].lower().find('diagnosis')!=-1 or lines[i+2][0].lower().find('diagnosis')!=-1:
            # n+=1
            # flag=1
            # break
        # else:
    out_line = ['Past Surgical History in hnp', filename, Time]
    out_line.extend(D)
    # ------------- into separate file
    out = open('profile/' + PID + '.csv', 'a', newline='', encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(out_line)
    # ------------- into separate file

    # csv_write.writerow([filename,Time,D])

    # if started==0:
    # print(filename)

print(n)
# csv_write.writerow([PID,Time])
