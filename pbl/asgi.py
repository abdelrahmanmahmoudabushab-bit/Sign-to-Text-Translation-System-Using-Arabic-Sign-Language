import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from app.consumers import SignLanguageConsumer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/live-translate/", SignLanguageConsumer.as_asgi()),
        ])
    ),
})
