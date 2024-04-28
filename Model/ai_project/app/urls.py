from django.urls import include, path
from rest_framework import routers
from .views import *

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('cifar', CifarClassification.as_view(), name='cifar'), # GET, POST, PATCH, PUT,...
    # path('cifar', Cifar10Classification.as_view(), name='cifar10'), # GET, POST, PATCH, PUT,...
    path('recommend', WebMining.as_view(), name='recommend'),
]
