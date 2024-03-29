from django import views
from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
   path('', views.index, name="home"),
   path('login/', views.signin, name="login"),
   path('logout/', views.logoutUser, name="logout"),
   path('signup/', views.signup, name="signup"),
   path('index', views.contact, name="contact"),
   path('trigger-flask-function/', views.trigger_flask_function, name='trigger-flask-function'),
   path('enroll', views.enroll, name="enroll"),

]
