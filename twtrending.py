"""
@author: chuquangnguyenhoang
"""
#Import lib
import os
from numpy import result_type
import tweepy
import configparser
import pandas as pd
from datetime import timedelta, date, datetime
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import RcParams
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import re
import time

def auth_api():
    # GET TWITTER'S KEYS, TOKENS AND AUTHORIZATION BY TWEEPY
    config = configparser.RawConfigParser()
    config.read('C:/Users/chuqu/Documents/Internship_project/Twitter_Sentiment_Analysis/dev/main code/key1.cfg')
    consumer_key1 = config.get('API', 'key1')                               # read api1, api2 from file
    consumer_secret1 = config.get('API', 'secret1')
    access_token1 = config.get('API', 'token1')
    access_token_secret1 = config.get('API', 'tokensecret1')
    consumer_key2 = config.get('API', 'key2')
    consumer_secret2 = config.get('API', 'secret2')
    access_token2 = config.get('API', 'token2')
    access_token_secret2 = config.get('API', 'tokensecret2')
    auth1 = tweepy.OAuthHandler(consumer_key1, consumer_secret1)            # get Auth by Tweepy
    auth2= tweepy.OAuthHandler(consumer_key2, consumer_secret2)
    auth2.set_access_token(access_token2, access_token_secret2)
    auth1.set_access_token(access_token1, access_token_secret1)
    api1 = tweepy.API(auth1, wait_on_rate_limit=True)
    api2 = tweepy.API(auth2, wait_on_rate_limit=True)
    return api1, api2


def get_trends(api, loc):
    """
    This function's used for getting trending topic on US's twitter

    Input:
        api - api address
        loc - location where you need to get trends (WOEID typed)
    """
    trends = api.get_place_trends(loc)
    return trends[0]["trends"]                                          # Take only trends's name


def scraping (numtweet, db):
    """
    Input:
        numtweet - number of tweets you want to scrape for each topic
        db - given dictionary for storing infromation you need to scrape (check main)
    """
    db = scrape(numtweet, api1, db, 0, 5)                               # scraping tweets by trending [0:5] using api1
    db = scrape(numtweet, api2, db, 5, 10)                              # scraping tweets by trending [5:10] using api2
    return db


def scrape(numtweet, api, db , a, b):
    """
    Input:
        numtweet - number of tweets you want to scrape for each topic
        api - api(1,2) address
        db - given dictionary for storing infromation you need to scrape (check main)
        a, b - order in top 10 trending topic ( for avoiding limit, 
        seperated into 2 [0:5] & [5:10] for two 2 APIs 
    """
    for word in d[a:b]:                                                 # Word from trending [0:5] and [5:10]
        print (word)
        query = word + ' -filter:retweets'                              # Trending topic + remove RT tweet
        tweets = tweepy.Cursor(
            api.search_tweets,
            query,
            lang = 'en',
            result_type = 'popular',
            tweet_mode = 'extended').items(numtweet)
        list_tweets = [tweet for tweet in tweets]
        a=[]                                                            # Create a list for storing tweet's id str
        for tweet in list_tweets:
            id_str= tweet.id_str
            a.append(id_str)
        if len(a) is not numtweet or not list_tweets:                   # If len(list_tweet[popular]) is not enough
                                                                        # add tweet[mixed typed] till satisfies needs
            print('result with mixed')                                  
            tweets=tweepy.Cursor(api.search_tweets,
            q = query,
            lang = 'en',
            result_type = 'recent',
            tweet_mode = 'extended').items(300-len(list_tweets))        # 300
            for tweet1 in tweets:
                if tweet1.id_str not in a:                              # Checking duplicate tweet between popular and mixed typed
                    list_tweets.append(tweet1)
                    a.append(tweet1.id_str)
                if len(list_tweets) == numtweet:
                    break  
            print (len(list_tweets))                                    # Checking list_tweets's shape =30 ?
        if len(list_tweets) is not numtweet or not list_tweets:         # If len(list_tweet[popular]) is not enough
                                                                        # add tweet[mixed typed] till satisfies needs
            print('result with no result_type')                                  
            tweets=tweepy.Cursor(api.search_tweets,
            q = query,
            lang = 'en',
            tweet_mode = 'extended').items(numtweet)                    
            list_tweets.clear()
            list_tweets = [tweet for tweet in tweets]
            print (len(list_tweets))                                    # Checking list_tweets's shape =30 ?
        for tweet in list_tweets:                                       # Appending section to db dictionary
            username = tweet.user.screen_name
            text = tweet.full_text
            text = text.replace('\n', '').replace(',', '')              # Replace page break and ',' avoiding disorder the data in csv file 
            db['trendings'].append(word)
            db['username'].append(username)
            db['text'].append(text)
    return db


def roberta (df, r):
    """
    Input:
        df - Dataframe after scraping tweets
        r - Length of Dataframe
    """
    model= "cardiffnlp/twitter-roberta-base-sentiment"                  # Adress model for scoring tweet's sentiment
    tokenizer = AutoTokenizer.from_pretrained(model)                    # Get tokenize's model
    model = AutoModelForSequenceClassification.from_pretrained(model)   # Get scoring model 
    tokenizer.save_pretrained(model)
    text = df['text'].apply(lambda x: preprocess(x))
    robe_score(text, tokenizer, model, df, r )                          # Append sentimental score to df (def robe_score)
    robertafunc()                                                       # If Roberta works property print "successfully" 
    return df


def preprocess(text):
    """
    This function's used for cleaning text
    Input:
        text - tweet's full text    
    """
    text = str(text)
    # Replace "@mention and #hashtag and punctuation"
    text = " ".join(re.sub("(@[A-Za-z0-9_,.;?!&$]+)|(#[0-9A-Za-z_,.;?!&$]+)|(\w+:\/\/\S+)|([,.;?!&$])|([\n])"," ",text).split())
    text = " ".join(re.sub("[']","",text).split())
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def robe_score(tw, tokenizer, model, df, r):
    """
    This function's used for appending result to df 
    after calculating sentiment by "bert_cal" function
    
    Input:
        tw - tweet's text
        tokenizer - tokenize model
        df - Dataframe after scraping
        r - length of df
    """
    a =[]
    b =[]
    d =[]
    for i in r :
        a.append(bert_cal(tw[i], tokenizer, model)[0])
        b.append(bert_cal(tw[i], tokenizer, model)[1])
        d.append(bert_cal(tw[i], tokenizer, model)[2])
    df['Neg'] = pd.Series(a)
    df['Neu'] = pd.Series(b)
    df['Pos'] = pd.Series(d)


def bert_cal(tw, tokenizer, model):
    """
    This function's used for calculating tweet's sentiment score

    Input:
        tw - tweet's text
        tokenizer - tokenize model
        model - scoring model
    """
    tokens = tokenizer.encode(tw, return_tensors='pt')
    result = model(tokens)
    point = softmax((result['logits'].detach().clone()))
    c = point.numpy()[0]
    return c


def robertafunc():
    print ("Roberta Model has run properly")


def plot(x,k,l):
    """
    Input:
        x - dataframe after groupby['trendings']
        k, l : range of dataset
    """
    colors= ['orangered', 'dodgerblue', 'limegreen']
    splot=x[k:l].plot.barh('trendings', width= 0.8, color=colors)
    plt.legend(loc=1, prop={'size': 30})
    splot.invert_yaxis()
    custom_xlim = (0, 100)
    splot.set(xlim=custom_xlim)
    splot.set_xlabel('percentage', fontsize=50, fontweight='bold', loc='right')
    splot.set(ylabel=None)
    splot.xaxis.label.set_size(50)
    splot.tick_params(axis='x', labelsize=60)
    splot.tick_params(axis='y', labelsize=50)
    plt.yticks(rotation=45, va='center')
    plt.gcf().set_size_inches(30, 50.33333333333333333333333333333333333)           # 16:9 ratio 
    splot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))             # Present percentage on x axis
    for container in splot.containers:                                              # Present value for each bar
        splot.bar_label(container, fmt='%.0f%%', size=50)
    plt.title('Sentiment Score of Top 10 US Twitter Trends on '+ timepresent, fontsize=60,fontweight= 'bold', x=0.4, y=1.05)

    
def tweeting_trendings():
    """
    This function's used for tweet the result and images
    URL: "https://twitter.com/UTrendingbot"
    """
    images = (f'{fullname2}1_5.png', f'{fullname2}6_10.png')
    msg1 = f'Top 10 US Twitter\'s trending on {timepresent} with Sentiment Score \n no.1: {x.trendings[0]} \n no.2: {x.trendings[1]} \n no.3: {x.trendings[2]} \n no.4: {x.trendings[3]} \n no.5: {x.trendings[4]} \n no.6: {x.trendings[5]} \n no.7: {x.trendings[6]} \n no.8: {x.trendings[7]} \n no.9: {x.trendings[8]} \n no.10: {x.trendings[9]}' 
    msg2 = f'Top 10 US Twitter\'s trending on {timepresent} with Sentiment Score \n no.1: {x.trendings[0]} \n no.2: {x.trendings[1]} \n no.3: {x.trendings[2]} \n no.4: {x.trendings[3]} \n no.5: {x.trendings[4]} \n for more information : https://twitter.com/explore/tabs/trending/ '
    try: 
        media_ids = [api1.media_upload(i).media_id_string for i in images]
        api1.update_status(status=msg1,media_ids=media_ids)
    except Exception as e:
        print(e)
    except:
        media_ids = [api1.media_upload(i).media_id_string for i in images]
        api1.update_status(status=msg2,media_ids=media_ids)
    else:
        print('Upload Successful')


# Main
if __name__ == '__main__':
    api1, api2 = auth_api()
    start_time = time.time()
    db = {'trendings': [], 'username': [], 'text': []}
    loc = "23424977"                                                                 # US's WOEID
    trends = get_trends(api1, loc)
    d= [i['name'] for i in trends[0:10]]                                             # Take 10 of US's trendings list
    db = scraping(200, db)
    df = pd.DataFrame.from_dict(db)
    
    dt = date.today().strftime("%b-%d-%Y")
    date_time = datetime.today().strftime("%Y%m%d-%H%M")
    timepresent = date.today().strftime("%b-%d-%Y")
    
    newpath1 = 'data/' + dt
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
    fullname1 = os.path.join(newpath1, date_time+"tweetsfromtrending#s.csv")
    
    print("Scraping Successful")
    r = range(len(df['text']))
    df = roberta(df,r)
    df.to_csv(fullname1, index=False, mode="w+", encoding = 'utf-8')
    e = {'Neg':'mean', 'Neu':'mean', 'Pos':'mean'}                                   # Get [Neg, Neu, Pos]'s mean result from trendings by each tweets   
    x = df.groupby('trendings', sort=False , as_index=False).agg(e)                  # Groupby df database by "trendings" and [Neg, Neu, Pos]
    x[['Neg', 'Neu','Pos']]= x[['Neg','Neu','Pos']].apply(lambda x: x*100)
    
    newpath2 = 'fig/' + dt
    if not os.path.exists(newpath2):
        os.makedirs(newpath2)
    fullname2 = os.path.join(newpath2, date_time+"tweetsfromtrending#s.csv")           # Covert [Neg, Neu, Pos] to percentage unit
    
    plot_1 = plot(x,0, 5)
    plt.savefig(f'{fullname2}1_5.png', bbox_inches='tight')
    plot_2 = plot(x, 5, 10)
    plt.savefig(f'{fullname2}6_10.png', bbox_inches='tight')
    print("saved fig")
    tweeting_trendings()
    end_time = time.time()
    print("Duration: {}".format(end_time - start_time))                               # Checking proccessing's time
