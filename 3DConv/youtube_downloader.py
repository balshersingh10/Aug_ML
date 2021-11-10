from pytube import YouTube
import pandas as pd
import os
from pathlib import Path

df = pd.read_csv("./data/URLs.csv",index_col  = "id")

print(df.head())

SAVE_PATH = "./videos/"
short_links = df['trailer_url']
link = ["https://www.youtube.com/watch?v="+x for x in short_links]
print(len(link))
c = 0
for i in link:
	my_file = Path("./videos/"+str(c)+".mp4")
	if not my_file.is_file():
		print("Downloading "+ str(c))
		try:
			yt = YouTube(i)
		except:
			print("Connection Error")
		try:
			out = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download(SAVE_PATH)
			os.rename(out,"videos/"+str(c)+".mp4")
		except:
			print("Some Error!")
	c+=1
print('Task Completed!')
