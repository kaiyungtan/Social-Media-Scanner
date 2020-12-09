from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
import nltk
import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import joblib
 

app = Flask(__name__)


# load vectorizer_transform & pos_vectorizer_transform

#vectorizer = joblib.load('./model/vectorizer_transform.pkl')
#pos_vectorizer = joblib.load('./model/pos_vectorizer_transform.pkl')

# load model for predicting hate speech
nlp_model = joblib.load('./model/nlp_model.pkl')

# load vectorizer_transform & pos_vectorizer_transform
#vectorizer = pickle.load(open('./model/vectorizer_transform.pkl', 'rb'))
#pos_vectorizer = pickle.load(open('./model/pos_vectorizer_transform.pkl', 'rb'))

# load model for predicting hate speech
#nlp_model = pickle.load(open('./model/nlp_model.pkl', 'rb'))


# route to home
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


# route to direct predict house price by postal code
@app.route('/predict',methods=['POST'])
def predict():
        df = pd.read_csv("./data/labeled_data.csv")
        tweets=df.tweet

        # Feature generation
        stopwords=nltk.corpus.stopwords.words("english")

        other_exclusions = ["#ff", "ff", "rt"]
        stopwords.extend(other_exclusions)

        stemmer = PorterStemmer()


        def preprocess(text_string):
            """
            Accepts a text string and replaces:
            1) urls with URLHERE
            2) lots of whitespace with one instance
            3) mentions with MENTIONHERE

            This allows us to get standardized counts of urls and mentions
            Without caring about specific people mentioned
            """
            space_pattern = '\s+'
            giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            mention_regex = '@[\w\-]+'
            parsed_text = re.sub(space_pattern, ' ', text_string)
            parsed_text = re.sub(giant_url_regex, '', parsed_text)
            parsed_text = re.sub(mention_regex, '', parsed_text)
            return parsed_text

        def tokenize(tweet):
            """Removes punctuation & excess whitespace, sets to lowercase,
            and stems tweets. Returns a list of stemmed tokens."""
            tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
            tokens = [stemmer.stem(t) for t in tweet.split()]
            return tokens

        def basic_tokenize(tweet):
            """Same as tokenize but without the stemming"""
            tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
            return tweet.split()

        vectorizer = TfidfVectorizer(
                                    tokenizer=tokenize,
                                    preprocessor=preprocess,
                                    ngram_range=(1, 3),
                                    stop_words=stopwords,
                                    use_idf=True,
                                    smooth_idf=False,
                                    norm=None,
                                    decode_error='replace',
                                    max_features=10000,
                                    min_df=5,
                                    max_df=0.75
                                    )

        tfidf = vectorizer.fit_transform(tweets).toarray()
        vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores

        #Get POS tags for tweets and save as a string
        tweet_tags = []
        for t in tweets:
            tokens = basic_tokenize(preprocess(t))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            tweet_tags.append(tag_str)

        #We can use the TFIDF vectorizer to get a token matrix for the POS tags
        pos_vectorizer = TfidfVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 3),
            stop_words=None,
            use_idf=False,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=5000,
            min_df=5,
            max_df=0.75,
            )

        #Construct POS TF matrix and get vocab dict
        pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
        pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

        # Get other features

        sentiment_analyzer = VS()

        def count_twitter_objs(text_string):
            """
            Accepts a text string and replaces:
            1) urls with URLHERE
            2) lots of whitespace with one instance
            3) mentions with MENTIONHERE
            4) hashtags with HASHTAGHERE

            This allows us to get standardized counts of urls and mentions
            Without caring about specific people mentioned.
            
            Returns counts of urls, mentions, and hashtags.
            """
            space_pattern = '\s+'
            giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            mention_regex = '@[\w\-]+'
            hashtag_regex = '#[\w\-]+'
            parsed_text = re.sub(space_pattern, ' ', text_string)
            parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
            parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
            parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
            return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

        def other_features(tweet):
            """This function takes a string and returns a list of features.
            These include Sentiment scores, Text and Readability scores,
            as well as Twitter specific features"""
            sentiment = sentiment_analyzer.polarity_scores(tweet)
            
            words = preprocess(tweet) #Get text only
            
            syllables = textstat.syllable_count(words)
            num_chars = sum(len(w) for w in words)
            num_chars_total = len(tweet)
            num_terms = len(tweet.split())
            num_words = len(words.split())
            avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
            num_unique_terms = len(set(words.split()))
            
            ###Modified FK grade, where avg words per sentence is just num words/1
            FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
            ##Modified FRE score, where sentence fixed to 1
            FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
            
            twitter_objs = count_twitter_objs(tweet)
            retweet = 0
            if "rt" in words:
                retweet = 1
            features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                        num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                        twitter_objs[2], twitter_objs[1],
                        twitter_objs[0], retweet]
            #features = pandas.DataFrame(features)
            return features

        def get_feature_array(tweets):
            feats=[]
            for t in tweets:
                feats.append(other_features(t))
            return np.array(feats)

        if request.method == 'POST':
            text = request.form['message']
        
        text = pd.Series(text)
        tfidf = vectorizer.transform(text).toarray()

        #Get POS tags for tweets and save as a string
        tweet_tags = []
        for t in text:
            tokens = basic_tokenize(preprocess(t))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            tweet_tags.append(tag_str)

        pos = pos_vectorizer.transform(pd.Series(tweet_tags)).toarray()
        
        feats = get_feature_array(text)
        
        #Now join them all up
        X = np.concatenate([tfidf,pos,feats],axis=1)
        

        my_prediction = nlp_model.predict(X)
        
        if my_prediction[0] == 2:
            prediction = 'neutral.'
             
        elif my_prediction[0] == 1:
            prediction = 'offensive.'
                 
        else:
            prediction = 'hateful.'
             
        return render_template('result.html', prediction = 'This sentence is {} '.format(prediction))


# return result of the prediction
@app.route("/result")
def result():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)