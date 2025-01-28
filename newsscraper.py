import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import time

# Define news sources and their biases
news_sources = {
    "https://www.cnn.com/world": "left",
    "https://www.foxnews.com/world": "right",
    "https://www.npr.org/sections/news/": "center"
}

# Function to extract article links from a news site
def get_article_links(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            full_url = a["href"]
            if full_url.startswith("/"):  
                full_url = url.rstrip("/") + full_url
            if "http" in full_url and "202" in full_url:
                links.append(full_url)
        return list(set(links))[:5]  
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

# Function to extract article content
def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None

# Scrape articles and label them
data = []
for source, bias in news_sources.items():
    print(f"Scraping: {source}")
    article_links = get_article_links(source)
    for link in article_links:
        content = extract_article_content(link)
        if content:
            data.append({"source": source, "bias": bias, "url": link, "content": content})
        time.sleep(1)  

# Save dataset
df = pd.DataFrame(data)
df.to_csv("news_bias_dataset.csv", index=False)
print("Dataset saved as news_bias_dataset.csv")
