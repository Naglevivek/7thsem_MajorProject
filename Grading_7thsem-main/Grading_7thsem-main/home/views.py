from multiprocessing import context
from django.shortcuts import render, HttpResponse
from datetime import datetime
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from .forms import CreateUserForm
from home.models import Contact,Enroll
from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
import requests 
from django.http import HttpResponse

def trigger_flask_function(request):
    if request.method == 'POST':
        # Define the Flask project URL
        flask_url = "http://localhost:5000"  # Adjust the URL as needed

        # Make a POST request to the Flask endpoint
        response = requests.post(flask_url + '/trigger-flask-function')

        if response.status_code == 200:
            # Capture the Flask response and return it as an HttpResponse
            flask_response = response.text
            return HttpResponse(flask_response)
        else:
            return HttpResponse("Failed to trigger Flask function.")
    else:
        return HttpResponse("Invalid request method.")


# Create your views here.
def index(request):
    return render(request, 'index.html')

def logoutUser(request):
    logout(request)
    return redirect('home')

def signin(request):
            if request.method == 'POST':
                username = request.POST.get('username')
                password =request.POST.get('password')
                user = authenticate(request, username=username, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('home')
                else:
                    messages.info(request, 'Username OR password is incorrect')

            context = {}
            return render(request, 'login.html')

def signup(request):
            form = CreateUserForm()
            if request.method == 'POST':
                form = CreateUserForm(request.POST)
                if form.is_valid():
                    form.save()
                    user = form.cleaned_data.get('username')
                    messages.success(request, 'Account was created for ' + user)
                    return redirect('login')
            context = {'form':form}
            return render(request, 'signup.html',context= context)


def contact(request):
    if request.method =="POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        contact= Contact(name=name, email=email, message=message, date=datetime.today())
        contact.save()
        messages.success(request, 'Your message has been send.')
    return render(request,'index.html')

def enroll(request):
    if request.method =="POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        contact= Enroll(name=name, email=email, phone_number=phone_number, date=datetime.today())
        contact.save()
        messages.success(request, 'Your message has been send.')
    return render(request,'index.html')