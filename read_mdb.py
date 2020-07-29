from dataset import LmdbDataset, hierarchical_dataset, AlignCollate
import argparse
import string
import torch

def read_mdb(opt):
    nick = 'result/nickname'
    nick_val = 'result/nickname_val'
    print(f"------{nick}-------")
    # dataset = LmdbDataset(nick,opt)
    dataset, datasetlog = hierarchical_dataset(nick,opt)
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid, pin_memory=True)

    # for idx, data in enumerate(dataset):
    #     print(data)
    #     if idx==1:
    #         break

    for i, (image_tensors, labels) in enumerate(valid_loader):
        print(labels[0])
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=45, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')

    opt = parser.parse_args()

    charlist = []
    with open('ko_char.txt', "r", encoding = "utf-8-sig") as f:
        for c in f.readlines():
            charlist.append(c[:-1])
    number = '0123456789'
    symbol  = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
    en_char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    opt.character = number + symbol + en_char + ''.join(charlist)

    opt.data_filtering_off = True

    read_mdb(opt)