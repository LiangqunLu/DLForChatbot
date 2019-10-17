from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.template.loader import get_template

from .forms import NameForm

import os, sys
#BASE_DIR = '/home/llu/HardDisk/LiangqunLuGitHub/DLForChatbot/web/mysite'

sys.path.append( os.path.join(os.getcwd(), '../../python/') ) 
from GoT_generation import txt_generation
from Eng_CHN_translation import txt_translation
from txt_DS import DS_generation
from txt_DS_seq2seq import DS_generation
from WSD_WordNet import WSD_generation

import pandas as pd

# Create your views here.
def index(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    content = {}
    title = 'Text Generation'

    request.session['seed_text'] = ""
    request.session['output'] = ""
    request.session['true_seq'] = ""

    form = NameForm()

    if request.method == 'POST':

        form = NameForm(request.POST)

        if form.is_valid():

            seed_text, output, true_seq = txt_generation()

            request.session['seed_text'] = seed_text
            request.session['output'] = output
            request.session['true_seq'] = true_seq

            return render(request, 'DL/index.html', {'title': title, 'seed_txt': seed_text, 'output_txt': output, 'true_seq': true_seq  } )
    
    return render(request, 'DL/index.html', {'title': title} )


def translate(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    content = {}
    title = 'Text Translation'

    request.session['input_text'] = ""
    request.session['output'] = ""
    request.session['true_seq'] = ""

    form = NameForm()

    if request.method == 'POST':

        form = NameForm(request.POST)

        if form.is_valid():

            input_text, output, true_seq = txt_translation()

            request.session['input_text'] = input_text
            request.session['output'] = output
            request.session['true_seq'] = true_seq

            return render(request, 'DL/translate.html', {'title': title, 'input_txt': input_text, 'output_txt': output, 'true_seq': true_seq  } )
    
    return render(request, 'DL/translate.html', {'title': title} )


def DS(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    content = {}
    title = 'Dialogue Generation'

    request.session['input_text'] = ""
    request.session['output'] = ""
    request.session['true_seq'] = ""

    form = NameForm()

    if request.method == 'POST':

        form = NameForm(request.POST)

        if form.is_valid():

            input_text, output, true_seq = DS_generation()

            request.session['input_text'] = input_text
            request.session['output'] = output
            request.session['true_seq'] = true_seq

            return render(request, 'DL/DS.html', {'title': title, 'input_txt': input_text, 'output_txt': output, 'true_seq': true_seq  } )
    
    return render(request, 'DL/DS.html', {'title': title} )

def WSD(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    content = {}
    title = 'Word Sense Disambiguation'

    request.session['input_txt'] = ""
    request.session['output'] = ""
    request.session['true_seq'] = ""

    form = NameForm()

    if request.method == 'POST':

        form = NameForm(request.POST)

        if form.is_valid():

            #input_text, output, true_seq = DS_generation()
            #txt_g = form.cleaned_data['txt_g']
            txt_g = request.POST['input_sent']
            txt_w = request.POST['input_word']
            
            input_text, output, true_seq = WSD_generation(txt_g, txt_w)
            
            input_text = input_text
            output = output
            true_seq = true_seq
            
            request.session['input_text'] = input_text
            request.session['output'] = output
            request.session['true_seq'] = true_seq

            return render(request, 'DL/WSD.html', {'title': title, 'input_txt': input_text, 'output_txt': output, 'true_seq': true_seq  } )
        #else:
            #form = NameForm()
    
    return render(request, 'DL/WSD.html', {'title': title} )


def PTSD(request):
    
    content = {}
    title = 'PTSD articles'
    total_number = 4732
    #request.session['total_number'] = 4732
    data_df = pd.read_csv(os.path.join(os.getcwd(), '../../data/') + 'PTSD_Pubmed.txt', sep = "\t")   
    data_df = data_df.iloc[:, 1:]
    #request.session['total_number'] += data_df.shape[0]
    data_df = data_df.to_html() 

    return render(request, 'DL/PTSD.html', {'title': title, 'total_number': total_number , 'data': data_df } )
