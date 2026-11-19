from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/chart-insight/', views.api_chart_insight, name='api_chart_insight'),
    path('api/predict-insight/', views.api_predict_insight, name='api_predict_insight'),
    path('api/chart-ai/', views.api_chart_ai, name='api_chart_ai'),
]