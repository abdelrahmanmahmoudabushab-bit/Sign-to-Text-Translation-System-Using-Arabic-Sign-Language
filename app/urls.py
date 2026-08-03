from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('smooth_sentence/', views.smooth_sentence, name='smooth_sentence'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('api/telemetry/', views.api_telemetry, name='api_telemetry'),
    path('api/clear_logs/', views.api_clear_logs, name='api_clear_logs'),
    path('api/new_signs/', views.api_new_signs, name='api_new_signs'),
]
