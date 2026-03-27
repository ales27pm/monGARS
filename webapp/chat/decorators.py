from functools import wraps

from django.shortcuts import redirect

from .session_access import session_get


def require_token(view_func):
    @wraps(view_func)
    async def _wrapped(request, *args, **kwargs):
        token = await session_get(request, "token")
        uid = await session_get(request, "user_id")
        if not token or not uid:
            return redirect("login")
        request.token = token
        request.user_id = uid
        return await view_func(request, *args, **kwargs)

    return _wrapped
