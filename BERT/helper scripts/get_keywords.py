import urllib.request
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
from keybert import KeyBERT

df = pd.read_csv("./history.csv")

urls = df['url']
kw_model = KeyBERT()

for url in urls:
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        soup = BeautifulSoup(html,'html5lib')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        print(kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None))
    except Exception as e:
        print(e)
