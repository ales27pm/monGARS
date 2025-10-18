from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.login_view, name="login"),
    path("user/list/", views.admin_user_list, name="admin-user-list"),
    path(
        "user/change-password/",
        views.admin_change_password,
        name="admin-change-password",
    ),
]
