import shutil
import os

target = "./dataset"
genres = os.listdir(target)

for g in genres:
    print("Finalizing "+g)
    target_dir = os.path.join(target,g)
    file_dir = target_dir
    frames = os.listdir(target_dir)
    count = 0
    for f in frames:
        print(f,end=" ")
        source_dir = os.path.join(target_dir,f)
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            file =  str(count) + file_name
            count+=1
            file_dir = os.path.join(target_dir,file)
            shutil.move(os.path.join(source_dir, file_name), file_dir)
        os.rmdir(os.path.join(os.path.join(target,g),f))
    print("Done")
