from django.shortcuts import render
from rest_framework import viewsets, generics, status
from django.core.mail import send_mail
from rest_framework.pagination import PageNumberPagination
from rest_framework.viewsets import ModelViewSet
from django.views.decorators.csrf import csrf_exempt
# from datetime import date, time, timedelta
# import hashlib
from rest_framework.views import APIView
from rest_framework.response import Response
# import face_recognition
import datetime
from django.conf import settings
# import cv2
# from keras_vggface import utils
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io
# from django.http import JsonResponse
import numpy as np
# import shutil
# import re
# import string
# import random
# from django.db.models import Count
# from rest_framework import generics
# from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# import calendar

# cifar classification
import keras as keras
from rest_framework.parsers import MultiPartParser
from keras.models import load_model
import joblib
from keras.preprocessing.image import img_to_array, load_img
import cv2
from PIL import Image
# cifar10 classification
import tensorflow as tf
# from .models import mobileNetV2


class CifarClassification(APIView):
    parser_classes = [MultiPartParser]
    loaded_best_model = joblib.load('./models/best_logistic_regression_model.pkl')
    VGG16_base_model = load_model('./models/VGG16_base_model.h5')
    label_names = ['Bluebell', 'Buttercup', 'ColtsFoot', 'Cowslip', 'Crocus', 'Daffodil', 'Daisy', 'Dandelion', 'Fritillary', 'Iris', 'LilyValley', 'Pansy', 'Snowdrop', 'Suncifar', 'Tigerlily', 'Tulip', 'Windcifar']

    def get(self, request):
        data = {'message': 'GET request received!'}
        return Response(data, status=status.HTTP_200_OK)
    
    def post(self, request):
        if 'image_input' not in request.data:
            return Response({'error': 'No image file found'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Đọc hình ảnh từ request và chuyển đổi thành mảng numpy
        image_data = request.data['image_input']
        image_pil = Image.open(image_data)
        image_np = np.array(image_pil)

        # Resize hình ảnh sử dụng OpenCV
        image_resized = cv2.resize(image_np, (224, 224))

        # Tiếp tục xử lý hình ảnh như bình thường
        image = img_to_array(image_resized) 
        image = np.expand_dims(image, 0) 
        feature = self.VGG16_base_model.predict(image) 
        feature = feature.reshape((feature.shape[0], 512*7*7)) 
        pred = self.loaded_best_model.predict(feature) 
        response_data = {
            "data": {
                "cifars_name":self.label_names[pred[0]]
            },
            "messages": [
                "Successful cifar identification !"
            ],
            "status": 200
        }
        return Response(response_data, status=status.HTTP_201_CREATED)
    
# class Cifar10Classification(APIView):
#     cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     def get(self, request):
#         data = {'message': 'GET request received!'}
#         return Response(data, status=status.HTTP_200_OK)
#     def post(self, request):
#         if 'image_input' not in request.data:
#             return Response({'error': 'No image file found'}, status=status.HTTP_400_BAD_REQUEST)
        
#         # Đọc hình ảnh từ request và chuyển đổi thành mảng numpy
#         image_data = request.data['image_input']
#         image_pil = Image.open(image_data)
#         image_np = np.array(image_pil)

#         # Resize hình ảnh sử dụng OpenCV
#         image_resized = cv2.resize(image_np, (28, 28))

#         # Tiếp tục xử lý hình ảnh như bình thường
#         img_array = np.array(image_resized) / 255.0
#         img_array = np.expand_dims(img_array, 0) 
#         y_predict = mobileNetV2.predict(img_array)
#         pred = self.cifar10_label_names[np.argmax(y_predict)]
#         response_data = {
#             "data": {
#                 "cifar10_name": pred
#             },
#             "messages": [
#                 "Successful cifar10 identification !"
#             ],
#             "status": 200
#         }
#         return Response(response_data, status=status.HTTP_201_CREATED)

class WebMining(APIView):
    def post(self, request):
        id_user = request.data['id_user'] # POST 
        # id_user = request.data.get('id_user') # GET 
        recommend_products = [1,2,3,4,5,9,9,9,9]
        data = {
            'message': 'Get list product recommend success !',
            'recommend_products': recommend_products,
            'id_user': id_user,
        }
        return Response(data, status=status.HTTP_200_OK)
    
    