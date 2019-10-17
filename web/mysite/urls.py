"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
#from django.urls import include, path
from django.conf.urls import url, include

## this url file is not actually used, instead the mysite/url.py is used to load apps

urlpatterns = [
    #path('admin/', admin.site.urls),
    url(r'^$', include('DL.urls') ),
    path('', include('DL.urls')),
    path('DL/', include('DL.urls')),
    path('PTSD/', include('PTSD.urls')), 
    url('chat/', include('chat.urls') ),
    path('polls/', include('polls.urls')),
    path('admin/', admin.site.urls),

]
