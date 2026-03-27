from pathlib import Path

from django.conf import settings
from django.contrib import admin
from django.urls import re_path
from django.urls import include, path
from django.views.static import serve as static_serve

_STATIC_DOCUMENT_ROOT = (
    settings.STATIC_ROOT
    or (settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else None)
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("chat/", include("chat.urls")),
]

if _STATIC_DOCUMENT_ROOT:
    urlpatterns.append(
        re_path(
            r"^static/(?P<path>.*)$",
            static_serve,
            {"document_root": str(Path(_STATIC_DOCUMENT_ROOT))},
        )
    )
