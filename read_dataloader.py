import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import CustomInferenceDataset, AlignCollate
from model import Model
from detectimg import detectimg
from detection import get_detector
from imgproc import morphological_transformation

from PIL import Image

from glob import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def read_dataloader(opt):
    # load OCR model
    DETECTOR_PATH = 'craft_mlt_25k.pth'
    detector = get_detector(DETECTOR_PATH, device)
    print("Load Model Successfully")
    # Make Dataset
    label_list = []
    img_list = []
    coord_list = []
    for image_dir in opt.image_folder:
        # detect image
        image_list, max_width = detectimg(image_dir,detector)
        coord_list.append(image_list[0][0])
        img_list.append(image_list[0][1])
        label_list.append((image_dir[image_dir.find('/',6)+1:image_dir.find('.')]).replace(" ",""))
    
    # if(opt.morphological):
    #     img_list = morphological_transformation(img_list)
    print(f"Len img_list and label list {len(img_list), len(label_list)}")
    print("Make list Successfully")
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = CustomInferenceDataset(image_list=img_list, label_list=label_list, opt=opt)  # use InferenceDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    print("Make Dataloader Successfully")
    print(f"len dataset {len(demo_data)}")
    print(f'len datalader {len(demo_loader.dataset)}')

    with torch.no_grad():
        for image_tensors, label in demo_loader:
            # Image.fromarray(image_tensors.numpy()).show()
            print(len(label))
            print(label)
            print("*")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--infer_num',required=True,type=int,help='number of inference data')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--stat_dict', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--morphological', action='store_true', help='applay image morphological transformation')
    
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        charlist = []
        with open('ko_char.txt', 'r', encoding='utf-8') as f:
            for c in f.readlines():
                charlist.append(c[:-1])
        opt.character = ''.join(charlist) + string.printable[:-38]

    if opt.stat_dict:
        opt.character = '0123456789'
        # charlist = []
        # with open('ko_char.txt', "r", encoding = "utf-8-sig") as f:
        #     for c in f.readlines():
        #         charlist.append(c[:-1])
        # number = '0123456789'
        # symbol  = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        # en_char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # opt.character = number + symbol + en_char + ''.join(charlist)


    opt.image_folder = glob(opt.image_folder + '*')
    
    if len(opt.image_folder) > opt.infer_num:
        opt.image_folder = opt.image_folder[:opt.infer_num]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    read_dataloader(opt)