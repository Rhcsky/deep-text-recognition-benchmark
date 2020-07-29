from tqdm import tqdm
import os


with open('data_nick/gt.txt', 'r', encoding='utf-8') as data:
    datalist = data.readlines()

nSamples = len(datalist)
for i in tqdm(range(nSamples)):
    imagePath, label = datalist[i].strip('\n').split('\t')
    print(label)
    enc = label.encode()
    print(enc)
    dec = enc.decode('utf-8')
    print(dec)
    print(dec.replace(' ',''))
    if i==0:
        break