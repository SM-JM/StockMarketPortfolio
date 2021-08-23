# -*- coding: utf-8 -*-
"""ScrapeInitialData.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pxVpCa-7Gr4zLY02nKIsaJpM23mtrSQ6
"""

pip install requests

"""# Imports"""

import requests
import os
import pandas as pd
from bs4    import BeautifulSoup

"""# Mount Google Drive """

# Commented out IPython magic to ensure Python compatibility.
currentWorkingDir = !pwd
defaultWorkingDir = "/content"

if ( currentWorkingDir[0] == defaultWorkingDir ):
  from google.colab import drive

  drive.mount('/content/drive')
      
#   %cd "/content/drive/My Drive/Colab Notebooks/stock_portfolio"
else:
  print("Currenting running app from: ")
  !pwd

"""# Extract stock discloure information posted between 2016-2020

## Get list of entities
"""

df_listed     = pd.read_csv("listed-companies-main.csv")
df_listed_jnr = pd.read_csv("listed-companies-jnr.csv")

df_listed

df_listed_jnr

df_listed[df_listed['Type']=="ORDINARY"].InstrumentCode

df_listed_jnr[df_listed_jnr['Type']=="ORDINARY"].InstrumentCode

listedComp_lst = df_listed[df_listed['Type']=="ORDINARY"].InstrumentCode.tolist()
listedComp_lst.extend(df_listed_jnr[df_listed_jnr['Type']=="ORDINARY"].InstrumentCode.tolist())

len(listedComp_lst)

pd.DataFrame(listedComp_lst,columns=["Symbol"]).to_csv('listed_companies.csv')

postid_lst          = []
title_lst           = []
link_lst            = []
post_date_lst       = []
instument_code_lst  = []
content_lst         = []

for c in listedComp_lst:
  Base_URL  = 'https://www.jamstockex.com/tag/'
  URL = Base_URL + c

  articles_on_page_in_time_range_flag = True
  # Articles are arranged in decending order such that most recent articles are
  # on page 1



  while articles_on_page_in_time_range_flag:
    
    soup = BeautifulSoup(requests.get(URL).content, 'html.parser')

    for article in soup.findAll({'article'}):
      link = article.find('a')
      
      # Print article id
      print(article['id'])
      postid_lst.append(article['id'])

      # Print article title, link
      print(link.string)
      title_lst.append(link.string)
      print(link.get('href'))
      link_lst.append(link.get('href'))

      # Print post date
      post_date = article.find('p', attrs={'class':'text-muted'}).string

      # Print Instrument code
      print(c)
      instument_code_lst.append(c)

      import datetime
      post_date_formatted = post_date.string.replace("Posted: ","")
      post_date_formatted = post_date_formatted.replace(" at", "")
      
      ## Convert date string "post_date" to a format that can be used by strptime
      ## This involves ensuring that the hour characters are padded with 0.

      hour_index = post_date_formatted.find(":") - 2

      if (int(post_date_formatted[hour_index:hour_index+2])) <= 9:
        def insert_zero(string, index):
          return string[:index] + '0' + string[index:]
        post_date_formatted = insert_zero(post_date_formatted,hour_index+1)

      # Print formatted post date
      dt = datetime.datetime.strptime(post_date_formatted, "%B %d, %Y %I:%M %p")
      print(dt)
      post_date_lst.append(dt)

      # Print article content
      content_div = article.find('div', attrs={'class':'entry-content'})
      if content_div != None:
        if content_div.p != None:
          if content_div.p.string != None:
            content_summary = content_div.p.string
          else:
            content_summary = ""
        else:
            content_summary = ""
      else:
        content_summary = ""

      ## Determine if content is a summary by checking for ellipsis string: "[...]"
      ellipsis_string = "[\u2026]"

      if content_summary[-3:] != ellipsis_string:
        print(content_summary)
        content_lst.append(content_summary)
      else:
        print("**Extraction of full text needed**")
        URL_full_article  = link.get('href')
        soup_full_article = BeautifulSoup(requests.get(URL_full_article).content, 'html.parser')

        content_div_full_article=(soup_full_article.find('div', attrs={'class':'entry-content'}))
        
        content_str = ""
        for p in content_div_full_article.findAll({'p'}):
          if p.string != None:
            content_str = content_str + '\n' + p.string
            if len(p.string) >= 100:
              print(p.string)
              break
        content_lst.append(content_str)

      print("")
      if dt <= datetime.datetime(2015,12,31):
        articles_on_page_in_time_range_flag = False
        print("**Exit Scrape, date too old**")

    # Get link for next page
    element = soup.find('a', attrs={'class': "nextpostslink"})
    
    if element != None:
      URL = element.get('href')
    else:
      ## If link for next page can't be found, data extraction complete, so exit loop
      articles_on_page_in_time_range_flag = False

df = pd.DataFrame({
    'postid': postid_lst,
    'post_date': post_date_lst,
    'instrument_code':instument_code_lst,
    'title':title_lst,
    'link':link_lst,
    'content': content_lst,
    'content_length': map(len,content_lst),
})

df.to_csv('articles.csv')

df = pd.read_csv('articles.csv')

df

"""## Generate statistics """

df.shape

len(df.instrument_code.unique())

df.groupby(['instrument_code']).mean()

df.groupby(['instrument_code']).count()

df.describe()

df.to_csv('articles.csv',index=False)

"""# Generate statistics on trading information"""

df_trading = pd.read_csv('price-history.csv')

df_trading

df_trading_filtered = df_trading[df_trading['Symbol'].isin(listedComp_lst)]

df_trading_filtered

df_trading_filtered.groupby(['Symbol']).mean()

df_trading_filtered.groupby(['Symbol']).count()

grouped = df_trading_filtered.groupby(['Symbol']).count().reset_index()

grouped.sort_values(['Last_Traded'],ascending=False)

grouped.to_csv('trading_count.csv')