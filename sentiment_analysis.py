import pandas as pd
import numpy as np
import re
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from algorithms import KMPSearch

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

def sentimentAnalysis(country_code, method="freq"):
    word_list, _ = article(country_code)

    if method == "freq":
        sentiment_info = compute_sentiment(word_list, country_code)
    elif method == "vader":
        sentiment_info = sentiment_scores_vader(word_list, country_code)
    else:
        raise Exception(f"Sentiment analysis method ({method}) doesn't exist. Please enter either \"freq\" or \"vader\".")
    
    return sentiment_info

def article(country_code):
    stop_words = nltk.corpus.stopwords.words('english')
    article_word_cnt = 0
    all_lists = []

    # convert the paragraphs into token
    def tokenize(text):
        tokens = re.split("\W+", text)
        return tokens

    #remove stopwords and empty strings from the list
    def removeStopwords(tokenized):
        filtered = [i for i in tokenized if i not in stop_words]
        while " " in all_lists:
            all_lists.remove(" ")
        return filtered

    for i in range(5):
        df = pd.read_csv(f"./content/{country_code} {i+1}.txt", delimiter='\t', header=None)
        df.columns = ['Paragraphs']

        # calculate total word count of article
        for j in range(len(df.index)):
            article_word_cnt += len(df["Paragraphs"][j])
        
        df['Tokenized message'] = df['Paragraphs'].apply(lambda x: tokenize(x.lower()))

        # get filtered word list
        df['Filtered'] = df['Tokenized message'].apply(lambda x: removeStopwords(x))

        
        for j in range(len(df.index)):
            all_words = [paragraphs for paragraphs in df['Filtered']]
            all_lists += all_words[j]

    return all_lists, article_word_cnt

def compute_sentiment(word_list, country_code):
    def positive(word_list):
        pos = open("./content/Positive.txt", encoding="utf8")
        pos = pos.read().splitlines()
        
        pos_count = 0
        for x in range(len(word_list)):
            for y in range(len(pos)):
                pos_count = pos_count + KMPSearch(pos[y], word_list[x])[0]
                
        pos_per = (pos_count/(len(word_list))*100)
        
        return pos_per, pos_count

    def negative(word_list):
        neg = open("./content/Negative.txt", encoding="utf8")
        neg = neg.read().splitlines()
        
        neg_count = 0
        for x in range(len(word_list)):
            for y in range(len(neg)):
                neg_count = neg_count + KMPSearch(neg[y], word_list[x])[0]
                
        neg_per = (neg_count/(len(word_list))*100)
        
        return neg_per, neg_count

    def neutral(word_list, pos_count, neg_count):
        neu_count = len(word_list) - pos_count - neg_count
        neu_per = (neu_count / (len(word_list)) * 100)
        return neu_per, neu_count

    sentiment_info = {}
    
    pos_per, pos_count = positive(word_list)
    neg_per, neg_count = negative(word_list)
    neu_per, neu_count = neutral(word_list, pos_count, neg_count)
    
    sentiment_info["country_code"] = country_code
    sentiment_info["pos_per"] = pos_per
    sentiment_info["neg_per"] = neg_per
    sentiment_info["neu_per"] = neu_per
    
    if pos_per > neg_per:
        sentiment_info["overall"] = "positive"
    elif pos_per < neg_per:
        sentiment_info["overall"] = "negative"
    else:
        sentiment_info["overall"] = "neutral"
    
    sentiment_info["score"] = pos_per / (pos_per + neg_per)

    return sentiment_info


def sentiment_scores_vader(word_list, country_code):
    # dictionary to keep the output
    sentiment_info = {}

    # Convert list of words into sentence
    sentence = " ".join([str(x) for x in word_list])
    
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    vader_sentiment_dict = sid_obj.polarity_scores(sentence)

    sentiment_info["country_code"] = country_code
    sentiment_info["pos_per"] = vader_sentiment_dict["pos"] * 100
    sentiment_info["neg_per"] = vader_sentiment_dict["neg"] * 100
    sentiment_info["neu_per"] = vader_sentiment_dict["neu"] * 100
    
    # decide sentiment as positive, negative and neutral
    if vader_sentiment_dict['compound'] >= 0.05 :
        sentiment_info["overall"] = "positive"
 
    elif vader_sentiment_dict['compound'] <= - 0.05 :
        sentiment_info["overall"] = "negative"
 
    else :
        sentiment_info["overall"] = "neutral"

    sentiment_info["score"] = sentiment_info["pos_per"] / (sentiment_info["pos_per"] + sentiment_info["neg_per"])
    
    return sentiment_info

def display_wordcloud(word_list, country_code):
    wordcloud = WordCloud(width = 1000, height = 1000,
                          min_font_size = 10).generate(str(word_list))
    plt.figure(figsize=(4,4), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.title(country_code)
    plt.show()

def display_pie_chart(pos_per, neg_per, neu_per, overall_sent, sent_score, country_code):
    #plot pie chart of each countries
    plt.pie([pos_per, neg_per, neu_per], explode=(0, 0.2, 0.2), labels=["Positive", "Negative", "Neutral"], autopct="%1.1f%%")
    plt.title(f"Sentiment Analysis of {country_code}\nOverall sentiment = {overall_sent}, Sentiment score = {round(sent_score, 3)}")
    plt.show()

def plot_freq_vader_comparison(freq_sentiment_info, vader_sentiment_info, country_code):
    n_groups = 3
    percentage_freq = [freq_sentiment_info["pos_per"], freq_sentiment_info["neg_per"], freq_sentiment_info["neu_per"]]
    percentage_vader = [vader_sentiment_info["pos_per"], vader_sentiment_info["neg_per"], vader_sentiment_info["neu_per"]]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, percentage_freq, bar_width,
    alpha=opacity,
    color='b',
    label='Freq')

    rects2 = plt.bar(index + bar_width, percentage_vader, bar_width,
    alpha=opacity,
    color='g',
    label='Vader')

    plt.xlabel('Word Type')
    plt.ylabel('Percentage')
    plt.title(f"Comparison between Freq and Vader for {country_code}\nVader's score = {round(vader_sentiment_info['score'], 3)}, Freq's score = {round(freq_sentiment_info['score'], 3)}\nVader's overall = {vader_sentiment_info['overall']}, Freq's overall = {freq_sentiment_info['overall']}")
    plt.xticks(index + bar_width, ('Positive', 'Negative', 'Neutral'))
    plt.legend()

    plt.tight_layout()
    plt.show()