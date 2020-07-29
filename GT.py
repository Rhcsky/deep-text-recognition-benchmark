from glob import glob
import fire

def createGT():
    dirs = glob('test/*')
    start_idx = 5

    print(len(dirs))
    print(dirs[0])
    print(dirs[0][start_idx:])

    with open('gt.txt','w',encoding='utf-8') as gt:
        for i in dirs:
            gt.write('test/' + i[start_idx:] + '\t' + i[start_idx:-4] + '\n')

    print('Done')

if __name__ == '__main__':
    createGT()