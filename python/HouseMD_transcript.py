#!/home/llu/anaconda3/bin/python
#Transcript data: http://clinic-duty.livejournal.com

import os, sys
from importlib import reload
import path
import time
import re
import string
from collections import defaultdict
from operator import itemgetter
import textblob
from textblob import TextBlob
from urllib.request import urlopen
from bs4 import BeautifulSoup
import time

sys.path = [''] + sys.path
from python_modules import scraper, transcript

#episode_index_url = 'http://clinic-duty.livejournal.com/tag/episode%20index'
#all_raw_episodes = scraper.Scraper('../data/HouseMD_data/', episode_index_url).get_all_episodes()

all_raw_episodes = defaultdict(dict)
#html = urlopen("../data/HouseMD_data/episode_index.html").read().decode('utf-8')
html = open("../data/HouseMD_data/episode_index.html", "r").read()
soup = BeautifulSoup(html, 'html.parser')
div = soup.find("div", {"class": "entryText s2-entrytext "})

epi = [one.get_text() for one in div.findAll('b') if re.search('\d', one.get_text())]
title = [one.get_text() for one in div.findAll('a') if 'SEASON' not in one.get_text()]
title = [one for one in title if one != ' ']

def obtain_epi_script(epi_title = 'Pilot', epi_number = '1.01'):

    epi_div = div.find(text = epi_title, href=True)['href']    
    filename = '../data/HouseMD_data/' + epi_number + '_' + epi_title + '.txt'

    if not os.path.exists(filename):
        html = urlopen(epi_div).read().decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        script = soup.find("div", {"class": "entryText s2-entrytext "})    
        script = list(script.strings)

        disclaimer = [i for i, j in enumerate(script) if 'DISCLAIMER' in j][0]
        #end = [i for i, j in enumerate(script) if 'END' in j][0]
        #end = max(end, len(script) - 1 )

        script = script[(disclaimer + 1):(len(script) - 1)]

        out = open(filename, 'w')
        out.write('\n'.join(script))
        out.close()    
    else:
        script = open(filename, 'r').read().split('\n')

    return(script)        

#download txt 
all_episodes = defaultdict()
for a,b in zip(epi, title):

    epi_script = obtain_epi_script(epi_title = b, epi_number = a)
    season = a.split('.')[0]
    if season not in all_episodes:
        all_episodes[season] = {}
    else:
        all_episodes[season][a + '_' + b] = epi_script

    print('finish ' + a + '_' + b )
    time.sleep(3)

#merge txt
ffout = open('../data/HouseMD_data/alltranscript.txt', 'w')

for i in all_episodes.keys():
    for j, k in all_episodes[i].items():

        ffout.write('\n'.join(k))

ffout.close()







