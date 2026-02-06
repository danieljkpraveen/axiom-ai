from django.contrib.auth import views as auth_views
from django.urls import path

from . import views
from .forms import LoginForm

urlpatterns = [
    path("", views.index, name="chat_home"),
    path("chat/<uuid:session_id>/", views.session_view, name="chat_session"),
    path("api/chat/session/<uuid:session_id>/messages/", views.chat_messages_partial, name="chat_messages_partial"),
    path("api/chat/send/", views.chat_send, name="chat_send"),
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="auth/login.html", authentication_form=LoginForm),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("signup/", views.signup_view, name="signup"),
]
