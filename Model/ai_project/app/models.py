from django.db import models
from .MobileNet import MobileNetV2

# Create your models here.
def load_cifar10_model():
    mobileNetV2 = MobileNetV2(num_classes = 10)
    mobileNetV2.build((1, 32, 32, 3))
    mobileNetV2.summary()
    mobileNetV2.load_weights('./models/my_model_weights.h5')
    return mobileNetV2

# mobileNetV2 = load_cifar10_model()