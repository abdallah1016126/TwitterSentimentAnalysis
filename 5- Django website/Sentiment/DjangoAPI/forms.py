from django import forms
from .models import Tweet
from django.core.validators import MaxValueValidator, MinValueValidator

class TweetForm(forms.ModelForm):
    class Meta:
        model = Tweet
        fields = "__all__"

    DAY_CHOICES = (
    ('Sun', 'Sun'), ('Mon', 'Mon'), ('Tue', 'Tue'), ('Wed', 'Wed'), ('Thu', 'Thu'), ('Fri', 'Fri'), ('Sat', 'Sat'))
    text = forms.Textarea(attrs={'size':12,'title':'Tweet'})
    day = forms.TypedChoiceField(choices=DAY_CHOICES)
    hour = forms.IntegerField(validators=[
            MaxValueValidator(23),
            MinValueValidator(0)
        ])
    Suggested_Sentiment = forms.IntegerField(validators=[
            MaxValueValidator(1),
            MinValueValidator(0)
        ])
