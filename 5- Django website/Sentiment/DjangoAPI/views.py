from .forms import TweetForm
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .models import Tweet
from .serializer import TweetSerializers
from .preprocess import prerpocessText

import joblib
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
import math
import numpy as np
import random
from sklearn.utils import shuffle



class TweetView(viewsets.ModelViewSet):
    queryset = Tweet.objects.all()
    serializer_class = TweetSerializers

def status(data):
    try:
        final_model = joblib.load("E:\AUC\Courses\CSCE_4930 Selected Topics in CSCE (Intro to ML)\Project\Phase 5\Sentiment\DjangoAPI\model.joblib")
        y_pred= final_model.predict(data)
        y = np.argmax(y_pred, axis=1)
        result = "Positive" if y else "Negative"
        predpos = round(y_pred[0,1]*100,2)
        predneg = round(y_pred[0,0]*100,2)
        return [result, predpos, predneg]
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def update_weights():
    final_model = joblib.load("E:\AUC\Courses\CSCE_4930 Selected Topics in CSCE (Intro to ML)\Project\Phase 5\Sentiment\DjangoAPI\model.joblib")

    dayToIndex = {"Fri": 0, "Mon": 1, "Sat": 2, "Sun": 3, "Thu": 4, "Tue": 5, "Wed": 6}

    tweetsBeforeRetrain = 3

    queryset = Tweet.objects.all()
    if len(queryset) % tweetsBeforeRetrain == 0 and len(queryset) != 0:
        all_tweets = []
        sentimens=[]
        num = 0
        for entry in reversed(queryset):
            textDic = prerpocessText(entry.text)
            hour = entry.hour / 24.0
            days = np.zeros(7)
            days[dayToIndex[entry.day]] = 1

            l = []
            l.append(textDic)
            tweet = np.array(l)
            tweet = np.concatenate([tweet, days])
            tweet = np.concatenate([tweet, [hour]])
            all_tweets.append(tweet)

            sentimens.append(entry.Suggested_Sentiment)
            num += 1
            if num == tweetsBeforeRetrain:
                break

        all_tweets=np.array(all_tweets)
        sentimens = np.array(sentimens)
        final_model.fit((all_tweets, sentimens), batch_size=1, epochs = 2)

        #joblib.dump(final_model, "E:\AUC\Courses\CSCE_4930 Selected Topics in CSCE (Intro to ML)\Project\Phase 5\Sentiment\DjangoAPI\model.joblib", compress=True)



def FormView(request):
    if request.method == 'POST':
        form = TweetForm(request.POST or None)
        form.save()
        update_weights()

        if form.is_valid():
            text = form.cleaned_data['text']
            day = form.cleaned_data['day']
            hour = form.cleaned_data['hour']
            Suggested_Sentiment = form.cleaned_data['Suggested_Sentiment']
            # preprocess and call status
            textDic = prerpocessText(text)
            hour = hour/24.0

            dayToIndex={"Fri":0, "Mon":1, "Sat":2, "Sun":3, "Thu":4, "Tue":5, "Wed":6}
            days = np.zeros(7)
            days[dayToIndex[day]]=1

            l = []
            l.append(textDic)
            data = np.array(l)
            data = np.concatenate([data, days])
            data= np.concatenate([data, [hour]])
            ll = []
            ll.append(data)
            data=np.array(ll)
           # print(data)


            all_result = status(data)
            result = all_result[0]
            posres = all_result[1]
            negres = all_result[2]

            return render(request, 'status.html', {"data": result, "pospred":posres, "negpred":negres})

    form = TweetForm()
    return render(request, 'form.html', {'form': form})

