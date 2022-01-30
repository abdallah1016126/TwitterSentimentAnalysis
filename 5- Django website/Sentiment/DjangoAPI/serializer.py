from rest_framework import serializers
from .models import Tweet

class TweetSerializers(serializers.ModelSerializer):
    class meta:
        model = Tweet
        fields = '__all__'