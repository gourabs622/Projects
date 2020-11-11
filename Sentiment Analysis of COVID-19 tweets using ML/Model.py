# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import time
import sklearn
from random import shuffle
# nltk.download('vedar_lexicon')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Fetching Tweets from twitter
'''import tweepy
import csv #Import csv
auth = tweepy.auth.OAuthHandler('xxxxxxxxx', 'xxxxxxxxxxxx')
auth.set_access_token('xxxxxxxxxx', 'xxxxxxxxxxxx')

api = tweepy.API(auth,wait_on_rate_limit=True)

# Open/create a file to append data to
csvFile = open('Dataset.csv', 'a',encoding="utf-8",newline='')

#Use csv writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,count=100000,
                           q = "Covid-19",re.sub('[^a-zA-Z]', '',clean )
                           tweet_mode='extended',
                           lang = "en").items():

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.text,tweet.user.screen_name,tweet.user.location])
    #print(tweet.full_text,tweet.user.screen_name,tweet.user.location)
csvFile.close()'''

# Data Preprocessing
# Importing the Dataset
dataset = pd.read_csv('Dataset.csv', nrows = 10000)
dataset = dataset.drop('username', 1)
dataset = dataset.drop('location', 1)

# Data Cleaning
# Cleaning RT, Lnks and @Username
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
def clean_tweets(lst):
    lst = np.vectorize(remove_pattern)(lst, "\r")                                  
    lst = np.vectorize(remove_pattern)(lst, "\n")
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")                         
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")                            
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    return lst
dataset['text'] = clean_tweets(dataset['text'])

# Removing Emojis
def removeEmoji(result):
    return result.encode('ascii', 'ignore').decode('ascii')
dataset['text'] = [removeEmoji(i) for i in dataset['text']]

# Removing URL
def removeURL(str):
    temp = ''
    clean_1 = re.match('(.*?)http.*?\s?(.*?)', str)
    clean_2 = re.match('(.*?)https.*?\s?(.*?)', str)

    if clean_1:
        temp = temp + clean_1.group(1)
        temp = temp + clean_1.group(2)
    elif clean_2:
        temp = temp + clean_2.group(1)
        temp = temp + clean_2.group(2)
    else:
        temp = str
    return temp

dataset['text'] = dataset['text'].apply(lambda tweet: removeURL(tweet))

# Removing Punctuations
dataset['text'] = [re.sub('[^a-zA-Z]', ' ', i) for i in dataset['text']]

# Lowercase
dataset['text'] = [i.lower() for i in dataset['text']]

# Removing Extra-Whitespace
def remove_whitespace(mtext):
    return " ".join(mtext.split())
dataset['text'] = [remove_whitespace(i) for i in dataset['text']]

# Labeling the tweets with proper sentiment
import textblob
from textblob import TextBlob
def detect_sentiment(text):
    if TextBlob(text).sentiment.polarity > 0:
        return 1
    elif TextBlob(text).sentiment.polarity < 0:
        return -1
    else:
        return 0
dataset['sentiment'] = dataset.text.apply(detect_sentiment)

# Removing stopwords
stopWords = set(stopwords.words("english"))
dataset['text'] = dataset['text'].apply(lambda tweet: ' '.join([word for word in tweet.split() if word not in stopWords]))

# Stemming
ps = PorterStemmer()
focused_words = ['coronavirus', 'covid', 'quarantine', 'coronavirusoutbreak', 'virus', 'corona', 'lockdown', 'economy']
def stemWords(word):
    if word in focused_words:
        return word
    else:
        return ps.stem(word)

dataset['text'] = dataset['text'].apply(lambda tweet: ' '.join([stemWords(word) for word in tweet.split()]))

# Lematization
wnl = WordNetLemmatizer()

def lemmatizeWords(word):
    if word in focused_words:
        return word
    else:
        return wnl.lemmatize(word)

dataset['text'] = dataset['text'].apply(lambda tweet: ' '.join([lemmatizeWords(word) for word in tweet.split()]))

# Creating the Bag of Words Model
corpus = dataset['text']
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Model Processing
# Splitting the dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
'''from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()'''
classifier.fit(X_train, y_train)

# Prediction the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting the accuracy
from sklearn.metrics import classification_report, accuracy_score
print("The accuracy of this model is")
print(accuracy_score(y_pred,y_test)*100)
# The accuracy is 70% by using Naive Bayes and 88% by using Random Forest



'''
#--------------Plotting Section-------------

#-------Bargraph Of Sentiment--------

import seaborn as sns
see = []

for twt in dataset.sentiment:
    see.append(twt)

ax = sns.distplot(see,kde=False,bins=3)
ax.set(xlabel = 'Negative            Neutral             Positive'
       ,ylabel = '#Tweets',title = 'Sentiment Score of COVID-19 Tweets')




#-----------Emotions of the Tweets----------


from collections import Counter

def con(sentence):
    emotion_list = []
    sentence = sentence.split(' ')
    with open('emotions.txt','r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",",'').replace("'",'').strip()
            word, emotion = clear_line.split(':')

            if word in sentence:
                emotion_list.append(emotion)
        w = Counter(emotion_list)
        return w



dataset['emotion'] = dataset['text'].apply(lambda x: con(x))
emotions=con(dataset['text'].sum())



#--------Plotting Emotion Bargraph--------

plt.figure(figsize = (15,10))
plt.bar(emotions.keys(),emotions.values())
plt.xticks(rotation = 90)
plt.show()



#---------WordCloud-----------


from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator
import urllib
import requests
import matplotlib.pyplot as plt
def generate_wordcloud(all_words):
    Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
    image_colors = ImageColorGenerator(Mask)
    wc = WordCloud(background_color='black', height=750, width=2000,mask=Mask).generate(all_words)
    plt.figure(figsize=(10,20))
    plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
    plt.axis('off')
    plt.show()

#------Positve Sentiment--------

all_words = ' '.join([text for text in dataset['text'][dataset.sentiment == 1]])
generate_wordcloud(all_words)

#-----Negative Sentiment--------

all_words = ' '.join([text for text in dataset['text'][dataset.sentiment == -1]])
generate_wordcloud(all_words)

#------Neutral Sentiment--------

all_words = ' '.join([text for text in dataset['text'][dataset.sentiment == 0]])
generate_wordcloud(all_words)



#---------Pie Graph-----------

def percentage(part,whole):
    return 100*float(part)/float(whole)

positive = 0
negative = 0
neutral = 0
polarity = 0


for tweet in dataset.text:
    analyzer = TextBlob(tweet)
    polarity += analyzer.sentiment.polarity
    if analyzer.sentiment.polarity > 0:
        positive += 1
    elif analyzer.sentiment.polarity < 0:
        negative += 1
    else:
        neutral += 1
        
# print(positive)
# print(negative)
# print(neutral)
# print(polarity)

positive = percentage(positive,(positive + negative + neutral))
negative = percentage(negative,(positive + negative + neutral))
neutral = percentage(neutral,(positive + negative + neutral))

positive = format(positive,'.2f')
negative = format(negative,'.2f')
neutral = format(neutral,'.2f')

if (polarity > 0):
    print("Positive")
elif (polarity < 0):
    print("Negative")
elif (polarity == 0):
    print("Neutral")

labels = ['Positive ['+str(positive)+'%]', 'Negative ['+str(negative)+'%]', 
'Neutral ['+str(neutral)+'%]']
sizes = [positive, negative, neutral]
colors = ['lightskyblue','gold','lightcoral']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches,labels,loc="best")
plt.title("Polarity Pie Chart")
plt.axis('equal')
plt.tight_layout()
plt.show()



#-----------Tweets Location Bargraph----------

tweets['location'].value_counts().head(30).plot(kind='barh', figsize=(6,10))



#-------Fluctuations of Polarity---------

fig = plt.figure(figsize=(18,7))
sns.distplot(tweets['polarity'],kde=False)
plt.ylim(0,10000)



#-------Fluctuations of Subjectivity---------

fig = plt.figure(figsize=(18,7))
sns.distplot(tweets['subjectivity'],kde=False)
plt.ylim(0,10000)



#---------Creating Hastag FreqDist----------

# function to collect hashtags
def hashtag_extract(text_list):
    hashtags = []
    # Loop over the words in the tweet
    for text in text_list:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags

def generate_hashtag_freqdist(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 15 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 25)
    plt.figure(figsize=(16,7))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    plt.xticks(rotation=80)
    ax.set(ylabel = 'Count')
    plt.show()
    
hashtags = hashtag_extract(tweets['text'])
hashtags = sum(hashtags, [])

generate_hashtag_freqdist(hashtags)



#---------Plotting Confusion Matrix-----------
plt.figure(figsize = (10,7))
sns.heatmap(cm,annot=True)'''







