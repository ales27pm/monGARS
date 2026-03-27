from asgiref.sync import sync_to_async


async def session_get(request, key, default=None):
    return await sync_to_async(request.session.get, thread_sensitive=True)(
        key, default
    )


async def session_set(request, key, value) -> None:
    await sync_to_async(request.session.__setitem__, thread_sensitive=True)(
        key, value
    )


async def session_pop(request, key, default=None):
    return await sync_to_async(request.session.pop, thread_sensitive=True)(
        key, default
    )
