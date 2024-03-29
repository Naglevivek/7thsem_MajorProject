from django.contrib.auth.forms import UserCreationForm
from django.contrib.admin.widgets import AdminDateWidget
from django.contrib.auth.models import User
from django import forms
from django.forms import ModelForm


class CreateUserForm(UserCreationForm):
	username=forms.CharField(widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Your username"}))
	email=forms.CharField(widget=forms.EmailInput(attrs={"class":"form-control","placeholder":"Your email"}))
	password1=forms.CharField(widget=forms.PasswordInput(attrs={"class":"form-control","placeholder":"Your Password"}))
	password2=forms.CharField(widget=forms.PasswordInput(attrs={"class":"form-control","placeholder":"Confirm Password"}))
	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2']

# class ideaform(ModelForm):
# 	username=forms.CharField(widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Your username"}))
# 	email=forms.CharField(widget=forms.EmailInput(attrs={"class":"form-control","placeholder":"Your email"}))
# 	password1=forms.CharField(widget=forms.PasswordInput(attrs={"class":"form-control","placeholder":"Your Password"}))
# 	password2=forms.CharField(widget=forms.PasswordInput(attrs={"class":"form-control","placeholder":"Confirm Password"}))
# 	class Meta:
# 		model = idea
# 		fields = "__all__"

