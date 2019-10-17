from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index', views.index, name='index'),
    path('article', views.article, name='article'),
    
    path('general', views.general, name='general'),
    path('neuro', views.neuro, name='neuro'),
    path('genes', views.gene, name='genes'),       
    path('miRNAs', views.miRNA, name='miRNAs'),
    path('Data', views.miRNA, name='miRNAs'),
    path('metabolites', views.metabolites, name='metabolites'),

    path('search', views.search, name='search'),
    path('QA', views.QA, name='QA'),

]


