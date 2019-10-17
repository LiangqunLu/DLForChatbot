from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index', views.index, name='index'),
    path('translate', views.translate, name='translate'),
    path('DS', views.DS, name='DS'),
    #path('Models', views.Models, name='models'),
    path('WSD', views.WSD, name='WSD'),
    path('PTSD', views.PTSD, name='PTSD'),
]





