from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.template.loader import get_template


# Create your views here.
from django.shortcuts import render

def index(request):
    return render(request, 'chat/index.html', {})