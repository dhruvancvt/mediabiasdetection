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

# Define regex patterns for identifying news articles (adjust for different sites)
ARTICLE_PATTERNS = [
    r"/\d{4}/\d{2}/\d{2}/",  # Matches URLs containing YYYY/MM/DD (common news format)
    r"/article/",  # Matches URLs with "article" in them
    r"/story/",  # Matches URLs with "story"
    r"/news/",  # Matches URLs under a /news/ section
]

# Function to check if a URL looks like a news article
def is_article_url(url):
    return any(re.search(pattern, url) for pattern in ARTICLE_PATTERNS)

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
            
            if is_article_url(full_url):  # Filter using regex patterns
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
        
        # Ensure valid content (skip short articles)
        if len(article.text) > 200:  
            return article.text
        else:
            print(f"Skipped short content: {url}")
            return None
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
