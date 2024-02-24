from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('input/', views.input_form, name='input_form'),
    path('process_input/', views.process_input, name='process_input'),
    path('eeg/', views.eeg_view, name='eeg_view'),
]
