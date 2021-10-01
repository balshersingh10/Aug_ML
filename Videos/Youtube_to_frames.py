import youtube_dl
import cv2

url="https://www.youtube.com/watch?v=BuRPc1c7TGQ" #The Youtube URL
ydl_opts={}
ydl=youtube_dl.YoutubeDL(ydl_opts)
info_dict=ydl.extract_info(url, download=False)

formats = info_dict.get('formats',None)
print("Obtaining frames")
for f in formats:
    if f.get('format_note',None) == '360p':
        url = f.get('url',None)
        cap = cv2.VideoCapture(url)
        x=0
        count=0
        while x<1000:
            ret, frame = cap.read()
            if not ret:
                break
            filename ="./Video1/"+str(x)+".png"
            x+=1
            cv2.imwrite(filename.format(count), frame)
            count+=30 #Skip 300 frames i.e. 10 seconds for 30 fps
            cap.set(1,count)
            if cv2.waitKey(30)&0xFF == ord('q'):
                break
        cap.release()
