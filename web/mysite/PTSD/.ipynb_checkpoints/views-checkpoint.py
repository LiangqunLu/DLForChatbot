from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.template.loader import get_template
import os, sys
import pandas as pd
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator

from .forms import NameForm



def index(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    content = {}
    title = 'PTSD'

   #request.session['total_number'] = 4732
    data_df = pd.read_csv(os.path.join('/home/llu/HardDisk/Bitbucket/nlp/data/'+ 'PTSD_Pubmed.txt'), sep = "\t")   
    data_df = data_df.iloc[:, 1:]
    # obtain Review in Publication Type  
    review = [ "Review" in one  for one in data_df.iloc[:, 1].tolist()]
    data_df = data_df.loc[review, :]
    data_df = data_df.sort_values(by = ['DP_Date of Publication'], ascending=False)
    data_df = data_df.iloc[:10, ]
    data_df = data_df.loc[:, ['TI_Title', 'DP_Date of Publication']]

    data_df = data_df.to_html(header=False, index=False, border = 0, index_names=False) 

    return render(request, 'PTSD/index.html', {'title': title, 'data':data_df } )

def article(request):
    #return HttpResponse("Hello, world. You're at the polls index.")

    content = {}
    title = 'PTSD articles'

    #request.session['total_number'] = 4732
    data_df = pd.read_csv(os.path.join('/home/llu/HardDisk/Bitbucket/nlp/data/'+ 'PTSD_Pubmed.txt'), sep = "\t")
    data_df = data_df.sort_values(by = ['DP_Date of Publication'], ascending=False)   
    data_df = data_df.iloc[:100, 1:]
    #request.session['total_number'] += data_df.shape[0]
    data_df = data_df.to_html(index=False) 

    paginator = Paginator(data_df, 25) # Show 25 contacts per page
    page = request.GET.get('page')
    contacts = paginator.get_page(page)

    #return HttpResponse("Hello, world. You're at the polls index.")
    return render(request, 'PTSD/PTSD_article.html', {'title': title, 'data':data_df, 'contr':contacts } )

def general(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    content = {}
    title = 'PTSD General'

    #return HttpResponse("Hello, world. You're at the polls index.")
    return render(request, 'PTSD/PTSD_general.html', {'title': title} )

def neuro(request):
    title = 'PTSD neurobiology'
    return render(request, 'PTSD/PTSD_neuro.html', {'title': title} )

def gene(request):
    title = 'PTSD Genes'
    return render(request, 'PTSD/PTSD_genes.html', {'title': title} )


def miRNA(request):
    title = 'PTSD miRNAs'
    return render(request, 'PTSD/PTSD_miRNAs.html', {'title': title} )

def metabolites(request):
    title = 'PTSD metabolites'
    return render(request, 'PTSD/PTSD_metabolites.html', {'title': title} )

def QA(request):
    title = 'PTSD QA'
    return render(request, 'PTSD/PTSD_QA.html', {'title': title} )
  
def search(request):

    title = 'PTSD search'

    request.session['data'] = ""

    form = NameForm()
    if request.method == 'POST':

        form = NameForm(request.POST)

        if form.is_valid():

            data_df = pd.read_csv(os.path.join('/home/llu/HardDisk/Bitbucket/nlp/data/'+ 'PTSD_Pubmed.txt'), sep = "\t")
            data_df = data_df.sort_values(by = ['DP_Date of Publication'], ascending=False)
            
            article = [ "Journal Article" in one  for one in data_df.iloc[:, 1].tolist()] 
            review = [ "Review" in one  for one in data_df.iloc[:, 1].tolist()]


            data_df = data_df.loc[review or article, :]
            data_df = data_df.sort_values(by = ['DP_Date of Publication'], ascending=False)
            data_df = data_df.iloc[:10, ]
            data_df = data_df.loc[:, ['TI_Title', 'DP_Date of Publication']]
                
            data_df = data_df.to_html(index=False) 

            data_df = "This is a test"

            return render(request, 'PTSD/search.html', {'title': title, 'data': data_df } )

    return render(request, 'PTSD/search.html', {'title': title} )

