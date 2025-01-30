import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import time
from urllib.parse import urljoin
import re

# Define news sources and their biases
news_sources = {
    "https://www.cnn.com/world": "left",
    "https://www.foxnews.com/world": "right",
    "https://www.npr.org/sections/news/": "center"
}

# Function to extract article links from a news site
def get_article_links(base_url):
    try:
        response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200:
            print(f"Failed to retrieve {base_url}, status code: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        for a in soup.find_all("a", href=True):
            full_url = urljoin(base_url, a["href"])  # Ensure absolute URL
            
            # Check for date-like patterns (YYYY/MM/DD)
            if re.search(r"/\d{4}/\d{2}/\d{2}/", full_url):
                links.add(full_url)

        return list(links)[:5]  # Limit to first 5 articles
    except Exception as e:
        print(f"Error scraping {base_url}: {e}")
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
        print(f"Fetching: {link}")
        content = extract_article_content(link)
        
        if content:
            data.append({"source": source, "bias": bias, "url": link, "content": content})
        time.sleep(1)  # Rate limiting to avoid bans

# Save dataset
df = pd.DataFrame(data)
df.to_csv("news_bias_dataset.csv", index=False)
print("Dataset saved as news_bias_dataset.csv")
