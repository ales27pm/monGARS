from functools import wraps

from django.shortcuts import redirect


def require_token(view_func):
    @wraps(view_func)
    async def _wrapped(request, *args, **kwargs):
        token = request.session.get("token")
        uid = request.session.get("user_id")
        if not token or not uid:
            return redirect("login")
        request.token = token
        request.user_id = uid
        return await view_func(request, *args, **kwargs)

    return _wrapped
