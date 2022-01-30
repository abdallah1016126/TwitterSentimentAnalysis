# Twitter-Sentiment-Analysis
We, Amer Elsheikh and Abdallah Abdelaziz, designed and implemented a fully working ML model 
to predict tweets' sentiment with a 79% accuracy as a group project in Fall 2021.
We first made a literature review and chose sentiment140 (1.6M tweets) dataset to work on, 
then we preprocessed the data and generated useful featurs.
Then, we experimented different Sklearn models on the data, and the best was a neural network model 
associated with a glove-twitter-200 word embedding based on context, which gave 76% accuracy.
Thus, we decided to implement a neural network from scratch but with our own word embedding as an embedding
layer part of the network (based on sentiment) which gave 79% accuracy.
Lastly, we made a simple Django website for users to interactevly use the model. It also supports online 
learning, where we ask the users to provide feedback for the prediction and retrain the model using new
data every 

# TwitterSentimentAnalysis
