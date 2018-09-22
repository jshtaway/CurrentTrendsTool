#import dependencies#import 
import re
import time
import requests
import tweepy
import json


#math/data
import numpy as np
import math
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
from wordcloud import WordCloud
import collections

import wikipedia as wp
from pygeocoder import Geocoder
from googletrans import Translator

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

from bs4 import BeautifulSoup
from datetime import datetime
from datetime import date, timedelta


#from config import consumer_key, consumer_secret, access_token, access_token_secret
from apikeys import twitterAccessToken as access_token
from apikeys import twitterAccessTokenSecret as access_token_secret
from apikeys import twitterConsumerKey as consumer_key
from apikeys import twitterConsumerSecretKey as consumer_secret
newsKey = '3327bb97745b4f24b875a9bbc9154270'


# Setup Tweepy API Authentication# Setup 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())




def parse_url( url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    listylist=[]
    for table in soup.find_all('table'):
        listylist.append(parse_html_table(table))
    return listylist

def parse_html_table( table):
    n_columns = 0; n_rows=0; column_names = []

    # Find number of rows and columns
    # we also find the column titles if we can
    for row in table.find_all('tr'):

        # Determine the number of rows in the table
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            n_rows+=1
            if n_columns == 0:
                # Set the number of columns for our table
                n_columns = len(td_tags)

        # Handle column names if we find them
        th_tags = row.find_all('th') 
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())

    # Safeguard on Column Titles
    if len(column_names) > 0 and len(column_names) != n_columns:
        raise Exception("Column titles do not match the number of columns")

    columns = column_names if len(column_names) > 0 else range(0,n_columns)
    df = pd.DataFrame(columns = columns,
                      index= range(0,n_rows))
    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            df.iat[row_marker,column_marker] = column.get_text()
            column_marker += 1
        if len(columns) > 0:
            row_marker += 1

    # Convert to float if possible
    for col in df:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

    return df

def getCountryLanguages():
    #TODO: Use the .apply to just change the table to one dialect. Imrpove language scope later.
    df = parse_url('https://www.infoplease.com/world/countries-world/languages-spoken-each-country-world')
    countryLanguages = df[0].rename(columns={0:'country',1:'language'}).set_index('country')
    countryLanguages['language'] = [re.sub('\d+|%|\(.*\)|\s','',i).split(',')[0].split(';')[0] for i in countryLanguages['language']]
    return countryLanguages


#returns hashtag, followers, following, text, geo, date
#cityCountry example: 'paris,france'
def SearchForData(search_term, nTweets, cityCountry='',radius=100, numDays=1):
    
    # Setup Tweepy API Authentication
    #auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    #auth.set_access_token(access_token, access_token_secret)
    #api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    
    #--- Calculate geocordinates from cityCountry --- --- ---- ---- --- --- --- --
    if cityCountry:
        geoConvertTries = 0
        while True:
            try:
                result = Geocoder.geocode(cityCountry)
            except Exception as error:
                #print('errrooooorrrrr: ',error.message)
                if 'OVER_QUERY_LIMIT' in str(error):
                    print('Encountered an error:{0}\nWaiting 20 seconds and trying again.'.format(error))
                    time.sleep(20)
                    if geoConvertTries>10:
                        print("Could not convert geo. returning empty list")
                        return []
                else:
                    print("Could not convert geo. returning empty list")
                    return []
            else:
                break
        geoConvertTries+=1
        # 34.0934,56.134,50mi
        coords = str(result[0].coordinates).replace('(','').replace(')','') + f',{radius}mi'
        coords=coords.replace(' ','')
        print(cityCountry, ": ", coords)
    #--- ---- ----- ---- ---- ---- ---- ---- --- ---- ---- --- ---- --- ---- --- --

    #--- grab tweets --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ----
    maxTweets = 10000; oldest_tweet = None; unique_ids = []; desiredTweets = [];nTweetsPerDay=nTweets/numDays
    for day,num in zip([str(date.today() - timedelta(i)).split()[0] for i in range(numDays)], range(1,numDays+1)):
        tweetsPerDay=[];tries=1
        while len(desiredTweets) < min(nTweetsPerDay*num , maxTweets/numDays):
            
            #--- determine whether to grab tweets by geo or not --- ---- --- ----- --
            while True:
                try:
                    if cityCountry:
                        tweetsPerDay = api.search(search_term, count=nTweetsPerDay, result_type="recent", max_id=oldest_tweet, geocode=coords, until=day)
                    else:
                        tweetsPerDay = api.search(search_term, count=nTweetsPerDay, result_type="recent", max_id=oldest_tweet, until=day)
                    time.sleep(2)
                except Exception as error:
                    print(error,'Trying again after 1 minute.')
                    time.sleep(60)
                else:
                    break
            
            #---- ----- ----- ---- ----- ---- ----- ---- ----- ---- ---- ---- ---- --

            #--- Dont go through an infinite loop trying to fill tweets that don't exist -----
            if len(tweetsPerDay['statuses'])==0:
                print(f'No tweets returned while searching for \'{search_term}\'',len(desiredTweets)\
                     ,' ',day)
                if desiredTweets==[]:
                    return pd.DataFrame([{'text':'', 'vader':0,
                                          'location':cityCountry, 'geo':coords,
                                          'hashtags':0, 'followers':0,
                                         'friends_count':0,'statuses_count':0,
                                          'created_at':datetime.strptime(str(date.today()), '%Y-%m-%d')}])
                return pd.DataFrame(desiredTweets)

            #--- Append relevent tweets to output listy list ---- --- ---- ---- ---- --- ---
            for tweet in tweetsPerDay['statuses']:
                # Append tweet_id to ids list if it doesn't already exist. This allows checking for duplicate tweets
                if tweet["id"] not in unique_ids :
                    unique_ids.append(tweet['id'])
                    if cityCountry:
                        desiredTweets.append({'text':tweet['text'], 'vader':analyzer.polarity_scores(tweet['text'])['compound'],
                                          'location':cityCountry, 'geo':coords,
                                          'hashtags':tweet['entities']['hashtags'], 'followers':tweet['user']['followers_count'],
                                         'friends_count':tweet['user']['friends_count'],'statuses_count':tweet['user']['statuses_count'],
                                          'created_at':datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S %z %Y')})
                    else:
                        desiredTweets.append({'text':tweet['text'], 'vader':analyzer.polarity_scores(tweet['text'])['compound'],
                                          'hashtags':tweet['entities']['hashtags'], 'followers':tweet['user']['followers_count'],
                                         'friends_count':tweet['user']['friends_count'],'statuses_count':tweet['user']['statuses_count'],
                                          'created_at':datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S %z %Y')})
                
                # Reassign the the oldest tweet (i.e. the max_id) subtract 1 so the previous oldest isn't included
                oldest_tweet = tweet["id"] - 1
            print(f'{day}, {cityCountry}: attempt {tries}, {len(desiredTweets)} tweets gathered.');tries+=1

    #--- Print sample tweet --- --- ---- ---- --- ---- ---- --- ---- ---- ---
    translator = Translator()
    try:
        print ('Sample Tweet:',translator.translate(desiredTweets[0]['text'], dest='en').text)
    except:
        print('there was an error translating sample tweet: ',desiredTweets[0]['text'])
    return pd.DataFrame(desiredTweets) if cityCountry else desiredTweets


def worldTweets(search_term,numTweets, numDays):
    #-- Get the most populated cities from wikipedia (Thank you wikipedia library!) --
    html = wp.page("List_of_cities_by_population_density").html().encode("UTF-8")
    worldCities = pd.read_html(html)[1]
    worldCities = worldCities.drop([2,3,4],axis=1)
    worldCities = worldCities.rename(columns={0:'city',1:'population',5:'density',6:'country'})
    worldCities = worldCities.iloc[1:]
    worldCities['population'] = [int(city.split('\xa0')[-1].split('[')[0].replace(',','')) for city in worldCities['population']]
    worldCities['density'] = [int(city.split('\xa0')[-1].split('[')[0].replace(',','')) for city in worldCities['density']]
    #--- ---- ----- ----- ---- ---- ----- ---- ---- ----- --- ---- ---- ----- ---- ---
    
    #--- Get tweets by Worlds most densily populated cities ---- --- ---- ---- --- ---
    translator = Translator()
    comparisons=pd.DataFrame(columns=['time density','sentiment'])
    cityCount = 10
    for index,row in worldCities.iterrows():
        #-- location --- ----- --- ----
        city,pop,density,country = row
        cityCountry = city+' , '+country

        #-- language conversion --- ---- --
        languagesDf = getCountryLanguages()
        if translateToLocalLanguage:
            try:
                translatedSearch = translator.translate(search_term, src='en', dest=languagesDf.loc[country,'language']).text
            except ValueError:
                print("could not translate ", languagesDf.loc[country,'language'])
                translatedSearch=search_term
                print('translated word: ',translatedSearch)
        else:
            translatedSearch=search_term
        #--- --- --- ---- ---- --- --- ---
        
        try:
            tweetsWorld = pd.concat([tweetsWorld, SearchForData(translatedSearch, numTweets, cityCountry, 100)], axis=0)
        except:
            tweetsWorld = SearchForData(translatedSearch, numTweets, cityCountry, 100)
        print('\n')
        time.sleep(4)
        #if cityCount==0:
        #    break
        #else:
        #    cityCount-=1
    return tweetsWorld

def usTweets(search_term, numTweets, numDays):
    #--- population per cities in United States --- ---- ---- --- ---- --- --- --- ---
    html = wp.page("List_of_United_States_cities_by_population_density").html().encode("UTF-8")
    UScities = pd.read_html(html)[1]
    UScities = UScities.drop([0,2,4,6,8],axis=1)
    UScities = UScities.rename(columns={1:'city',3:'state',5: 'land area (mi^2)',7:'density'})
    UScities = UScities.iloc[1:]
    #df['population']=[int(city.split('\xa0')[-1].split('[')[0].replace(',','')) for city in df['population']]
    UScities['density'] = [float(city.split('\xa0')[-1].split('[')[0].replace(',','')) for city in UScities['density']]
    UScities['land area (mi^2)']=[float(area.split('\xa0')[-1]) for area in UScities['land area (mi^2)']]
    #--- ---- ----- ----- ---- ---- ----- ---- ---- ----- --- ---- ---- ----- ---- ---
    
    
    
    #--- Add US Cities --- ---- ---- ---- ---- ---
    cityCount = 100;nottheseones = 0
    for index,row in UScities.iterrows():
        if nottheseones<0:
            nottheseones+=1
            pass
        #-- location --- ----- --- ----
        city,state,area,density = row
        cityState = state+' , '+city

        try:
            tweetsUS = pd.concat([tweetsUS, SearchForData(search_term=search_term, nTweets=numTweets, cityCountry=cityState,radius=max(area,5), numDays=numDays )], axis=0)
        except:
            tweetsUS = SearchForData(search_term=search_term, nTweets=numTweets, cityCountry=cityState,radius=max(area,5), numDays=numDays)
        
        print('\n')
        time.sleep(4)
        if cityCount==0:
            break
        else:
            cityCount-=1
    return tweetsUS, UScities


def GetTweetsByPopularCities(search_term, numTweets, translateToLocalLanguage = True, numDays = 1):
    #tweetsWorld = worldTweets()
    tweetsUS, UScities = usTweets(search_term, numTweets, numDays)
    #return tweetsUS
    tweetsUS['day']=[time.date() if time!= 0 else 0 for time in tweetsUS['created_at']]
    
    
    #Calculate time density of tweets per city
    cityData = tweetsUS.groupby(['location','geo','day']).mean()
    aveTimes={}
    for date in tweetsUS.groupby(['location','geo','day'])['created_at']:
        if date == None:
            aveTimes[cityTime]=None
        cityTime, timeseries = date
        (cityState,geo,timeobj)=cityTime
        #print(cit)
        [state,city]=cityState.split(',')
        city=city.lstrip().rstrip(); state=state.rstrip().lstrip()
        cityDensity=UScities.loc[UScities['city']==city].loc[UScities['state']==state].loc[:,'density'].values[0]
        #return timeseries
        try:
            aveTimes[cityTime]=np.mean([(b-a).seconds*100/cityDensity for a,b in zip(timeseries.tolist()[1:],timeseries.tolist()[0:-1])])
            if not aveTimes[cityTime]:
                aveTimes[cityTime]=0
        except:
            if len(timeseries.tolist())<2:
                aveTimes[cityTime]=None
        #print(f'{city}, {cityDensity}, {aveTimes[cityState]}')
    #KeyError: ('New Jersey , Guttenberg', datetime.date(2018, 6, 30))
    cityData['time between tweets']=[aveTimes[citytime] for citytime in cityData.index]
    
    #c_matrix = pd.pivot_table(news_data_new, "score", index=["name"], columns=["year_month"], aggfunc=np.mean)                          
    matrix=pd.pivot_table(cityData.loc[:,['location','day','vader']].drop(columns=['location', 'day']),'vader',index=['location'],columns=['day'],aggfunc=np.mean)
    plt.figure(figsize=(12,12))
    hm = heatmap(matrix, cmap='RdYlGn', annot=True)
    hm.set_title('City Tweet Sentiment Over Time')
    plt.savefig('/plots/cityTweetSentimentOverTime.png',dpi=100)
    plt.show()
    return cityData, tweetsUS


def getSentimentsByPopularUsers(target_term):
    # Get 100 recent tweets and create lists of tweets, sentiments, followers and retweets
    public_tweets = api.search(target_term, count=100, lang="en", result_type="recent")
    tweets = []
    sentiments = []
    followers = []
    retweets = []
    for tweet in public_tweets['statuses']:
        tweets.append(tweet['text'])
        followers.append(tweet['user']['followers_count'])
        retweets.append(tweet['retweet_count'])
        result = analyzer.polarity_scores(tweet["text"])
        sentiments.append(result['compound'])
    # Create a dataframe by combining all the lists    
    combined_df = pd.DataFrame({'Followers':followers,'Sentiments':sentiments, 'Retweets':retweets, 'Tweets':tweets})
    
    # Calculate average sentiment of top 10 users with most followers
    top_10_followers = combined_df.nlargest(10, 'Followers')
    top_10_followers_average = round(top_10_followers['Sentiments'].mean(),3)

    top_10_retweets = combined_df.nlargest(10, 'Retweets')
    top_10_retweets_average = round(top_10_retweets['Sentiments'].mean(),3)

    overal_average = round(combined_df['Sentiments'].mean(),3)
    
    # Print analysis
    print('Average overall sentiment is ' + str(overal_average))
    print('Sentiment of top 10 influential people is '+str(top_10_followers_average))
    print('Sentiment of top 10 retweeted tweets is '+str(top_10_retweets_average))

    if top_10_followers_average>overal_average:
        print('Top 10 influential people have more positive sentiments than general public about '+target_term)
    else:
        print('General public has more positive sentiments than top 10 influential people about '+target_term)
    
    #Draw bar chart for sentiments
    plt.figure(figsize= (15,10))    
    titles = ['Influential Users','Retweeted Most','Overall average']
    sent = [top_10_followers_average, top_10_retweets_average, overal_average]
    plt.bar(titles, sent)

    plt.title(f"Sentiment Analysis {time.strftime('%x %X')}")
    plt.ylabel("Tweet Polarity")
    plt.axhline(0, color = 'black')
    plt.savefig("/plots/output_9_1.png")
    #plt.ylim (-1.0,1.0)
    plt.show()
    
    # Draw scatter chart for sentiments, number of followers and number of retweets    
    # Remove users with top 20 follower count to allow more meaningful visualization
    combined_df1 = combined_df.nsmallest(80, 'Followers')
    plt.figure(figsize= (15,10))
    plt.scatter(combined_df1['Followers'], combined_df1['Sentiments'], color="lightskyblue", edgecolor="black"
                , label="Number of retweets", alpha=0.5, s=combined_df1['Retweets'], linewidths=1)
    plt.title(f"Sentiment Analysis based on Number of Followers {time.strftime('%x %X')}")
    plt.xlabel("Number of Followers")
    plt.ylabel("Tweet Polarity")
    plt.ylim (-1.0,1.0)

    lgnd=plt.legend(loc="upper right", title="Sentiment Analysis based on Number of Followers", scatterpoints=1)
    lgnd.legendHandles[0]._sizes = [50]
    lgnd.legendHandles[0]._sizes = [50]
    lgnd.legendHandles[0]._sizes = [50]
    plt.savefig("/plots/output_8_1.png")
    plt.show()
    
def bar_news(news_data):
    '''takes a DataFrame as an argument, 
    creates a bar chart "Overal Quantity of Articles per News Source"
    ''' 
    #group data by News Sources names
    group_news=news_data.groupby("name")
    group_count=group_news['published_at'].count().rename("count")

    df_group_count=pd.DataFrame(group_count)
    df_group_count.sort_values(by=['count'], inplace=True, ascending=False)
    df_group_count.reset_index(inplace=True)
    
    #create a bar chart to show quantity of articles for each news source 
    group_name=df_group_count['name']
    y_pos=np.arange(len(df_group_count['count']))
    plt.figure(figsize=(10, 7))
    plt.barh(y_pos, df_group_count['count'], tick_label=group_name, alpha=0.6, color="green")
    plt.title(f"Overal Quantity of Articles per News Source, 09/01/17-{time.strftime('%x')}")
    plt.xlabel("Number of Articles")
    plt.savefig("/plots/news_bar_chart.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    #add analysis for bar chart
    q=len(df_group_count['count'])-1
    name_min=df_group_count.iloc[q,0]
    min_q=df_group_count.iloc[q,1]
    
    name_max=df_group_count.iloc[0,0]
    max_q=df_group_count.iloc[0,1]
    print(f'Based on the analysis of "Overal Quantity of Articles per News Source" chart, we can notice that {name_max} published the maximum number of articles ({max_q}).' ) 
    print(f'{name_min} published the minimal quantity of articles ({min_q}) related to the given keywords.')
    
def compound_scores_heat_map(news_data_new):
    '''takes a DataFrame with compound scores as an argument,
    creates a heatmap chart for average compound scores.
    Since some news sources didn't publish articles on particular months,
    using pd.pivot_table gives us a 2D Series where missing values are replaced with NaN.
    '''
    c_matrix = pd.pivot_table(news_data_new, "score", index=["name"], columns=["year_month"], aggfunc=np.mean)                          
    c_matrix=c_matrix.round(2)
    
    #create a heatmap chart of the average compound scores
    fig = plt.figure(figsize=(12,12))
    hm = sns.heatmap(c_matrix, cmap='RdYlGn', annot=True)
    hm.set_title(f"Heatmap of Avg. Compound Scores, 09/01/17-{time.strftime('%x')}")
    plt.savefig("/plots/compound_scores_heat_map.png", dpi=100)
    plt.show()


def  compound_score_data(news_data):
    '''takes a DataFrame as an argument and calculates compound scores for articles description,
    returns a new DataFrame
    '''
    news_data_new=news_data.loc[:,['year_month', 'name', 'description']]
    for index, row in news_data_new.iterrows():
        text=news_data_new.loc[index,"description"]
        if text:
            results = analyzer.polarity_scores(text)
            news_data_new.loc[index,"score"]=round(results["compound"],2)
        else:
            news_data_new.drop(index, inplace=True)
    print(news_data_new.head())
    compound_scores_heat_map(news_data_new)

    
def count_articles_heat_map(news_data): 
    '''takes a DataFrame as an argument,
    creates a heatmap chart "Number of Articles per Month"
    Since some news sources didn't publish articles on particular months,
    using pd.pivot_table gives us a 2D Series where missing values are replaced with NaN.
    '''
    d_matrix = pd.pivot_table(news_data, "url", index=["name"], columns=["year_month"], aggfunc='count')                          
    d_matrix.head(20)
    fig = plt.figure(figsize=(12,12))
    cm = sns.heatmap(d_matrix, cmap='YlGnBu', annot=True, fmt='g')
    cm.set_title(f"Number of Articles per Month, 09/01/17-{time.strftime('%x')}")
    plt.savefig("/plots/count_values_heat_map.png", dpi=100)
    plt.show()

    
def news_api_data(user_input):
    '''takes an argument "user_input" as keywords to perform API calls to newsapi.org,
    returns a DataFrame with news data
    ''' 
   
    #in case we have several words as user's input, we can mark words or phrases that must appear with a + symbol
    user_input_list=user_input.split(" ")
    user_input_plus="".join(f'+{k}' for k in user_input_list)
    print(user_input_plus)

    url="https://newsapi.org/v2/everything"
    
    #specify parameters for sample, i.e results come from the particular news  resources
    sparams={
       'apiKey': newsKey,
       'q': user_input_plus,
       'language': 'en',
       'pageSize': 100,
       'from': '2017-06-01',
       'sortBy': 'publishedAt',
       'sources': 'abc-news, associated-press, bbc-news, breitbart-news, cbc-news, cbs-news, cnn,  fox-news, daily-mail, independent, nbc-news, newsweek, new-york-magazine, reuters, time, the-globe-and-mail, the-telegraph, the-washington-times, usa-today'
    }
       
    #make API request and retrieve totalResults value 
    responses=requests.get(url, params=sparams)
    
    st_code=responses.status_code   
    if st_code!=200: 
        print(f'API problem, code {responses.status_code}')
    
    news_info=responses.json()
    #print(responses.json())
    
    sample_total=news_info['totalResults']

    #if response containes more than 10000 articles, retrive information for 10000.
    #Otherwise API returns an error 419 - too many requests 
    if sample_total>10000:
        sample_total=10000
    print(f'The total number of results is {sample_total}')

    #calculate the nuber of pages to iterate through (numer of all results is diveded by 100 (i.e. records per page))
    x=min(80, math.ceil(sample_total/100))
    print(f'The number of pages is {x}')

    #create DataFrame for retrieved results
    news_data=pd.DataFrame(columns=["published_at","name", "author", "description", "url","year_month"])
    index=0

    #iterate through each page and store  results in the DataFrame
    for page in range(1,x+1):
        sparams['page']=page
        responses=requests.get(url, params=sparams)    
        news_info=responses.json()
        #print(f'processing page {page} of {x}')
        if 'articles' not in news_info:
            #print(f'skipping page {page} of {x}')
            continue
        for article in news_info['articles']:
            news_data.loc[index,"published_at"]=article["publishedAt"]
            news_data.loc[index,"name"]=article["source"]["name"]
            news_data.loc[index,"author"]=article["author"]
            news_data.loc[index,"description"]=article["description"]
            news_data.loc[index,"url"]=article["url"]
            news_data.loc[index,"year_month"]=article["publishedAt"][:7]
            index+=1

    #clean the data from duplicate records 
    news_data.drop_duplicates(subset=['published_at','name','description'], keep='first', inplace=True)

    news_data.to_csv("report.csv")
    #news_data.info()
    bar_news( news_data)
    compound_score_data(news_data)
    count_articles_heat_map(news_data)
    #except:
    #    print('Query limit reached. Skipping news analysis')
    
def getWordCloud(search_term):
    #searchfordata inputs: search_term, nTweets, cityCountry='',radius=100, numDays=1
    tweet=SearchForData(search_term,100)
    # Getting all hashtags in a list
    hashtags = []
    text=''
    for t in range(len(tweet)):
        for u in (tweet[t]['hashtags']):
            hashtags.append(u['text'])

    #print(hashtags)
    # Assigning the list to a string variable since wordcloud needs it in a string variable and can'y use list
    text=' '.join(hashtags)
    #print(text)
    # Using WordCloud library to create a Wordcloud chart
    wordcloud = WordCloud(background_color='white',max_words=len(text),max_font_size=40,relative_scaling=.15).generate(text)

    plt.figure(figsize=(7,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('/plots/WordCloud.png',dpi=500)
    plt.show()


def main(search_term):
    timeStart = time.time()
    
    getSentimentsByPopularUsers(search_term)
    news_api_data(search_term)
    getWordCloud(search_term)
    #GetTweetsByPopularCities inputs:        search_term, numTweets, translateToLocalLanguage = True, numDays = 1
    cityDataMain = GetTweetsByPopularCities(search_term, numTweets=100,translateToLocalLanguage= False, numDays = 6)
    
    print(f'Test took {(time.time()-timeStart)//60} minutes')
    return cityDataMain
