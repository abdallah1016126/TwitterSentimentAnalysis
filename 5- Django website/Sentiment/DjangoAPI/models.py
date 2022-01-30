from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

class Tweet(models.Model):
    DAY_CHOICES = (('Sun', 'Sun'), ('Mon', 'Mon'), ('Tue', 'Tue'), ('Wed', 'Wed'), ('Thu', 'Thu'), ('Fri', 'Fri'), ('Sat', 'Sat'))
    text = models.TextField(max_length=280)
    day = models.CharField(max_length=3, choices=DAY_CHOICES)
    hour = models.IntegerField(validators=[
        MaxValueValidator(23),
        MinValueValidator(0)
    ])
    Suggested_Sentiment = models.IntegerField(validators=[
        MaxValueValidator(1),
        MinValueValidator(0)
    ])

    def __str__(self):
        return self.text