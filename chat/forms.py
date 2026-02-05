from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User


class SignupForm(UserCreationForm):
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"autocomplete": "username"}))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))

    class Meta:
        model = User
        fields = ("username", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault(
                "class",
                "w-full px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-800 text-neutral-100",
            )


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={"autocomplete": "username"}),
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={"autocomplete": "current-password"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault(
                "class",
                "w-full px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-800 text-neutral-100",
            )
