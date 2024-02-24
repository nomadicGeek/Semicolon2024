from django.urls import path

from . import views
from .views import upload_file



urlpatterns = [
    #path("", views.index, name="index"),
    #path('input/', views.input_form, name='input_form'),
    #path('process_input/', views.process_input, name='process_input'),
    path("", views.upload_file, name="upload_file"),
    path('eeg/', views.eeg_view, name='eeg_view'),
    path('upload', views.upload_file, name='upload_file'),
    path('suggestion/', views.suggestionsChatGPT, name='suggestionsChatGPT'),
    path('email/', views.sendPersonalizedEmail, name='sendPersonalizedEmail'),
    path('uploadecg/', views.upload_ecgfile, name='upload_ecgfile')
]
