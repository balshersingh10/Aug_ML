from keybert import KeyBERT
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import urllib.request
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
print("Imported dependencies")

df = pd.read_csv("./history.csv")
urls = df['url']

print("Loaded history successfully")
kw_model = KeyBERT()
print("KeyBERT loaded successfully")
docs = []
print("Keywords extraction in progress")
for url in urls:
    try:
        print("Requesting URL: ",url)
        req = Request(url, headers={'User-agent': 'your bot 0.1'})
        html = urlopen(req).read()
        soup = BeautifulSoup(html,'html5lib')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None)
        keywords = [a[0] for a in keywords]
        docs.extend(keywords)
    except Exception as e:
        print(e)
print("Keywords extracted successfully")
# Load sentence transformer model
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
print("SentenceTransformer setup done")
# Define UMAP model to reduce embeddings dimension
umap_model = umap.UMAP(n_neighbors=15,
                       n_components=10,
                       min_dist=0.0,
                       metric='cosine',
                       low_memory=False,
                       random_state = 42)

# Define HDBSCAN model to perform documents clustering
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10,
                                min_samples=1,
                                metric='euclidean',
                                cluster_selection_method='eom',
                                prediction_data=True)

# Create BERTopic model
topic_model = BERTopic(top_n_words=10,
                       n_gram_range=(1,2),
                       calculate_probabilities=True,
                       umap_model= umap_model,
                       hdbscan_model=hdbscan_model,
                       embedding_model=sentence_model,
                       verbose=True)

# Train model, extract topics and probabilities
topics, probabilities = topic_model.fit_transform(docs)
print("Topic model trained successfully")
num = topic_model.get_topic_freq()
print(len(num['Topic']))
for i in range(-1,len(num['Topic'])):
    print(topic_model.get_topic(i))
    print("\n")
#print(topic_model.get_topics())
#(topic_model.visualize_hierarchy())
