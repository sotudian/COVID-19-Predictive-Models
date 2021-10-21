from simple_sentence_segment import sentence_segment
from allennlp.predictors.predictor import Predictor
import scispacy
import spacy
import numpy as np
from unidecode import unidecode
import csv
import re
import os
csv.field_size_limit(1000000000)

nlp = spacy.load("en_ner_bc5cdr_md-0.2.4")

out = open('hnp_out_sym.csv', 'a', newline='', encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')

files = sorted(os.listdir('hnp/'))
# print(files)

n = 0

stop_list_disease = [
    'medications prior to admission',
    'prior to admission medication',
    'home medications',
    'allergies',
    'family history',
    'social history',
    'no past medical history on file',
    'medications',
    'surgical history',
    'ed course',
    'past medical history',
    'patient active problem',
    'course',
    'cardiology history',
    'PHYSICAL EXAMINATION'.lower(),
    'Exam',
    'ROS']
started = 0

predictor = Predictor.from_path(
    "mnli-roberta-large-2020.02.27.tar.gz",
    predictor_name="textual-entailment")
gender_table = [
    item for item in csv.reader(
        open(
            'Patient_list.csv',
            "r",
            encoding='utf-8'))]
gender_dic = {}

for nn in gender_table:
    gender_dic[nn[0]] = nn[1]


def combine_sentences(sentences):
    st = ''
    for i in range(len(sentences)):
        st += sentences[i]
        if i != len(sentences) - 1:
            st += ' '
    return st


def assert_stop(line, stop_list):
    a = 0
    for k in stop_list:
        if line[0].lower().find(k) != -1 or line[0].find(k) != -1:
            if line[0].find('past medical history') == -1:
                a = 1
                break
    return a


count = 0

for filename in files:
    nes_added = []
    nes_neg = []
    count += 1
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

    para = ''

    head = lines[0][0]
    # print(head.split('|'))
    PID = head.split('|')[0]
    Gender = gender_dic[PID]
    # print(PID)
    Time = head.split('|')[5]
    # print(Time)
    for (i, line) in enumerate(lines):
        if started == 0:
            if line[0].lower().find('chief complaint') != -1 or line[0].lower().find('reason for admission') != -1 or line[0].lower(
            ).find('HISTORY OF THE PRESENT ILLNESS'.lower()) != -1 or line[0].lower().find('History of Present Illness'.lower()) != -1:
                if line[0].find('chief complaint') == - \
                        1 and line[0].find('reason for admission') == -1:
                    # if line[0].lower().find('suthor')!=-1 or
                    # line[0].lower().find('reason for admission')!=-1:
                    started = 1
                    n += 1

        if started == 1:
            '''if line[0][0]=='?':
                    disease=line[0].split('    ')[0][1:].strip()
                    print(disease)
                    D.append(disease)'''

            if assert_stop(line, stop_list_disease) == 1:
                break
            para += line[0]
            para += ' '

            # if lines[i+1][0].lower().find('diagnosis')!=-1 or lines[i+2][0].lower().find('diagnosis')!=-1:
            # n+=1
            # flag=1
            # break
        # else:
    dic = {}
    # para=para.replace('_s','no')
    print(para)
    doc = nlp(para)
    nes_nlp = [i for i in list(doc.ents)]
    for ne in nes_nlp:
        if ne.label_ == 'DISEASE':
            # print(ne)
            dic[str(ne)] = 0
    nes = [
        'fever',
        'cough',
        'dyspnea',
        'shortness of breath',
        'SOB',
        'fatigue',
        'diarrhea',
        'loose stool',
        'nausea',
        'vomiting',
        'emesis',
        'abdominal pain',
        'abd pain',
        'loss of smell',
        'anosmia',
        'loss of taste',
        'chest pain',
        'headache',
        'sore throat',
        'hemoptysis',
        'bloody sputum',
        'myalgia',
        'muscle aches',
        'muscle pains']
    #nes=['SOB','abd pain']
    for ne in nes:
        # if ne.label_=='DISEASE':
            # print(ne)
        dic[str(ne)] = 0

    print(dic.keys())

    if Gender == 'Male':
        call = 'he'
    else:
        call = 'she'

    for k in dic:
        if para.lower().find('no ' + k.lower()) == -1 and para.lower().find('not ' + k.lower()) == -1 and para.lower().find('deny ' + k.lower()) == -1 and para.lower().find('denies ' + k.lower()) == - \
                1 and para.lower().find('denied ' + k.lower()) == -1 and para.lower().find('not ' + k.lower()) == -1 and para.lower().find('without ' + k.lower()) == -1 and para.lower().find('non ' + k.lower()) == -1:
            dic[k] = 1

    NES = [k for k in dic if dic[k] == 1]
    # print(NES)
    # if nes_nlp==[]:
    if para == '':
        csv_write.writerow([filename, Time])
        continue
    # -------------------------------TE task

    sentences = []
    for s, t in sentence_segment(para):
        sentence = para[s:t].strip().replace('\n', ' ')
        sentences.append(sentence)
    segments = []

    window = 1

    if len(sentences) <= window:
        segments.append(combine_sentences(sentences))
    else:
        for i in range(int(len(sentences) - window + 1)):
            segments.append(combine_sentences(sentences[i:int(i + window)]))

    for s in segments:
        for ne in NES:
            if ne in nes_added:
                continue
            if s.lower().find(ne.lower()) == -1:
                continue

            #p=predictor.predict(hypothesis=call+' has '+ne,premise=s)
            p = predictor.predict(hypothesis='has ' + ne, premise=s)
            if p['label'] == 'entailment':
                nes_added.append(ne)
            # if p['label']=='contradiction':
                # nes_neg.append(ne)

    nes_added = [i for i in nes_added if i not in nes_neg]
    nes_added = list(set(nes_added))
    print(nes_added)

    # ------------- into separate file
    '''out = open('profile/'+PID+'.csv', 'a', newline='',encoding='utf-8')
	csv_write = csv.writer(out, dialect='excel')
	csv_write.writerow([filename,'Disease in hnp',Time,D])'''
    # ------------- into separate file

    out_line = [filename, Time]
    out_line.extend(nes_added)
    csv_write.writerow(out_line)

    if count % 10 == 0:
        out.close()
        out = open('hnp_out_sym.csv', 'a', newline='', encoding='utf-8')
        csv_write = csv.writer(out, dialect='excel')

    # if started==0:
        # print(filename)

print(n)
# csv_write.writerow([PID,Time])
