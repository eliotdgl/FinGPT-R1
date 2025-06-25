"""
    == web_scraping_finance_news.py ==
    Quick script to get financial news from BusinessToday. 
"""
import requests as req
from bs4 import BeautifulSoup as BS

def get_news(url: str, max_news: int) -> list[str]:
  webpage = req.get(url)
  soup = BS(webpage.content, "html.parser")

  news = []
  nb = 1
  for link in soup.find_all('a'):
    if nb > max_news:
        break
    if (str(type(link.string)) == "<class 'bs4.element.NavigableString'>") and (link.get('href') is not None) and (len(link.string) > 35):
      news.append(link.string.strip())

  return news

# Financial news URLs
bt_url = "https://www.businesstoday.in/latest/economy" # BusinessToday

bt_news = get_news(bt_url, 50)
print(bt_news)
