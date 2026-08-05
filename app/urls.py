from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('smooth_sentence/', views.smooth_sentence, name='smooth_sentence'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('api/telemetry/', views.api_telemetry, name='api_telemetry'),
    path('api/telemetry/stats/', views.api_telemetry, name='api_telemetry_stats'),
    path('api/telemetry/events/', views.api_telemetry, name='api_telemetry_events'),
    path('api/clear_logs/', views.api_clear_logs, name='api_clear_logs'),
    path('api/new_signs/', views.api_new_signs, name='api_new_signs'),
    path('api/developer_logs/', views.api_developer_logs, name='api_developer_logs'),
    path('api/control_jetson/', views.api_control_jetson, name='api_control_jetson'),
    path('api/ollama_status/', views.api_ollama_status, name='api_ollama_status'),
]
