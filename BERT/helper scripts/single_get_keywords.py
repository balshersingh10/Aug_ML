import urllib.request
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
from keybert import KeyBERT

kw_model = KeyBERT()

url = "https://nochgames.com/gaming-blog-setup-ultimate-guide/"
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urlopen(req).read()

soup = BeautifulSoup(html,'html5lib')

# kill all script and style elements
for script in soup(["script", "style"]):
    script.decompose()    # rip it out

# get text
text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

#print(text)
print(kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None))
