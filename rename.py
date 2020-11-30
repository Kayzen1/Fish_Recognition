import os

def rename_files(fish):
    folder = './'+fish
    filelists = os.listdir(folder)
    for file in filelists:
        print(file)
        os.rename(os.path.join(folder,file),os.path.join(folder,fish+'_'+file))

fishs = []
# fishs = ['False Kelpfish','Filefish','Little Yellow Croaker','Soles','Squid']
for fish in fishs:
    rename_files(fish)