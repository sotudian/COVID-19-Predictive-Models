
import numpy as np
from unidecode import unidecode
import csv
import re
import os
csv.field_size_limit(1000000000)


cov_med = [
    'hydroxychloroquine',
    'ceftriaxone',
    'azithromycin',
    'remdesivir',
    'IVIG',
    'immunoglobulin',
    'steroids',
    'prednisone',
    'methylprednisolone']
htn_med = [
    item for item in csv.reader(
        open(
            'med_htn.csv',
            "r",
            encoding='utf-8'))]
htn_med = [i[0] for i in htn_med]
dia_med = [
    item for item in csv.reader(
        open(
            'med_dia.csv',
            "r",
            encoding='utf-8'))]
dia_med = [i[0] for i in dia_med]


def GenBanID():

    #BanList={'Pharmacologic Substance':0,'Biologically Active Substance':0,'Biologically Active Substance':0,'Biologically Active Substance':0}
    BanList = {'Pharmacologic Substance': 0, 'Clinical Drug': 0}
    Nonsense = {}

    Cate = {}
    SemDic = {}

    # f=open('MRDEF.RRF',encoding='utf-8')
    f = open('MRSTY.RRF', encoding='utf-8')

    f = f.readlines()
    print(f[1])

    for lines in f:
        line = lines.split('|')
        ID = line[0]
        cate = line[3]
        Cate[cate] = 0

        if cate in BanList:
            # print(line)
            Nonsense[ID] = 1

    for k in Cate:
        print(k)

    return Nonsense


TargetID = GenBanID()
print(TargetID)


ConDic = {}

# f=open('MRDEF.RRF',encoding='utf-8')
f = open('MRCONSO.RRF', encoding='utf-8')

f = f.readlines()
# print(f)
print(f[0])
print(f[1])
print(f[0].split('|'))
print(f[0].split('|')[0])
print(f[0].split('|')[14])

print(len(f))

for lines in f:
    line = lines.split('|')
    ID = line[0]
    Concept = line[14]

    if ID in TargetID:
        # ConDic[Concept.lower()]=1
        ConDic[Concept] = 1


print(ConDic['Medications'])


# ---------------------------------------------------------------------

out = open('vis_out_med_3.csv', 'a', newline='', encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')

files = sorted(os.listdir('visit/'))
# print(files)

n = 0

#stop_list_med=['allergies','family history','social history','no past medical history on file','surgical history','facility-administered medications','PHYSICAL EXAMINATION'.lower(),'Exam','ROS','EXAM','Vitals']
stop_list_med = [
    'allergies',
    'family history',
    'past medical history',
    'social history',
    'patient active problem',
    'no past medical history on file',
    'surgical history',
    'facility-administered medications',
    'PHYSICAL EXAMINATION'.lower(),
    'Exam',
    'ROS',
    'EXAM',
    'Vitals']
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


for filename in files:
    D = []
    covm = []
    print(filename)
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
    for (i, line) in enumerate(lines):
        if started == 0:
            # if line[0].lower().find('prior to admission medications')!=-1 or line[0].lower().find('medications prior to admission')!=-1:
            # if line[0].lower().find('current outpatient medications')!=-1:
            if line[0].find('Current Medications') != - \
                    1 or line[0].find('CURRENT MEDICATIONS') != -1:
                started = 1
                n += 1

        if started == 1:
            if assert_stop(line, stop_list_med) == 1:
                break
            if line[0][0] == '?':
                print(line)
                chem = line[0].split('  ')[0][1:].strip().split(' ')[0]
                chem_brand = re.findall(r'(\(.*?\))', line[0])
                chem_brand = [i[1:-1] for i in chem_brand]
                chem_brand.append(chem)
                print(chem_brand)
                for c in chem_brand:
                    if c in ConDic and c.lower() not in ['medications', 'medication'] and c not in [
                            'Level', 'I']:
                        D.append(line[0].split('  ')[0][1:].strip())
                        break
            else:
                chem = line[0].split('  ')[0].strip().split(' ')[0]
                chem_brand = re.findall(r'(\(.*?\))', line[0])
                chem_brand = [i[1:-1] for i in chem_brand]
                chem_brand.append(chem)
                print(chem_brand)
                for c in chem_brand:
                    if c in ConDic and c.lower() not in ['medications', 'medication'] and c not in [
                            'Level', 'I']:
                        D.append(line[0].split('  ')[0])
                        break

            for medi in cov_med:
                if line[0].lower().find(medi) != - \
                        1 or line[0].upper().find(medi) != -1:
                    covm.append(medi)

    Disease_inferred = []

    for m in dia_med:
        for dd in D:
            if dd.lower().find(m) != -1 or m.find(dd.lower()) != -1:
                if 'diabetes' not in Disease_inferred:
                    Disease_inferred.append('diabetes')

    out_line = ['Prior to Admission Medications in visit(5', filename, Time]
    out_line2 = ['COVID19 Medications in visit(5', filename, Time]
    out_line3 = ['Past Medical History inferred by medv(5', filename, Time]
    out_line.extend(D)
    out_line2.extend(covm)
    out_line3.extend(Disease_inferred)
    # ------------- into separate file
    out = open('profile/' + PID + '.csv', 'a', newline='', encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(out_line)
    csv_write.writerow(out_line2)
    csv_write.writerow(out_line3)
    # ------------- into separate file

    # csv_write.writerow([filename,Time,covm])
    # csv_write.writerow([filename,Time,D])
    # csv_write.writerow([filename,Time,Disease_inferred])

    # if started==0:
    # print(filename)

print(n)
# csv_write.writerow([PID,Time])
