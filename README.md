
# Media and Social Media Monitoring and Analytics.
Alena Dudko, Jennifer Shtaway, Rekha Amer, Shandiz Montazeri

## Project Overview

<p>Nowadays, advertisement departments and PR agencies are interested in the public response regarding their products. This is why companies such as Business Wire , NUVI, newsapi.aylien.com provide real-time social media  and media monitoring and analytics.</p>

<p>Our team decided to create a program that takes any user input as keywords, that they would like to analyze. Once a user provides a valid input, the program outputs graphs analyzing sentiment, location, timeline of tweet among other things. Besides, twitter API we also use News API to build graphs around user input to show media trends and tendencies.</p>


### Research Questions to Answer
<ul>
    <li>What are the related handles (hashtags) appearing together with the user searched term?</li>
    <li>What is the average sentiment of top-10 influential people (with the highest number of followers) about the target term ?</li>
    <li>What is the average sentiment of top-10 retweets about the target term ?</li>
    <li>What is the overall average sentiment about the target term ?</li>
    <li>Which news sources write mostly about the topic?</li>
    <li>What is an average sentiment score of their articles? How it changes over period of time?</li>
    <li> How the topic popularity  in media changes over period of time?</li>
    
</ul>
   


### Data Sources
Data sets  are dynamic and come from:
twitter API (api.search)
News API (newsapi.org)
Wikipedia
gmaps
Google Translate

### Project Steps:
    
<ul>
  <li>Data retrieving, selecting and cleaning.</li>
  <li>Collecting and grouping data in DataFrames.</li>
  <li>Creating pivot tables. Calculating averages, finding highest/lowest values.</li>
  <li>Building charts. Analyzing findings.</li>
</ul>

### Example Output w/ search term "SpaceX":
![](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/WordCloud.png)
![](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/compound_scores_heat_map.png)
![](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/count_values_heat_map.png)
![Sentement Analysis per Region](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/fig.png)
![](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/news_bar_chart.png)
![Sentement Analysis based on twitter user number of followers (Influencers)](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/output_8_1.png)
![](https://github.com/zen-gineer/CurrentTrendsTool/blob/master/plots/output_9_1.png)
