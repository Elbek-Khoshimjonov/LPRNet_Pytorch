from data.load_data import CHARS, CHARS_DICT, pre_process
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--test_img', type=str, required=True,  help='test image path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--pretrained_model', default='./weights/v3/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

# Run pre-processing, model and decode
def run(net, args, img):
    # Pre-process
    img = pre_process(img, args.img_size)
    img = torch.tensor(img)
    img = img.unsqueeze(0)

    img = Variable( img.cuda() if args.cuda else img)
    # Run model
    output = net(img).cpu().detach().numpy()
    
    return decode(output)

# Greedy decode
def decode(prebs):

    # output labels
    labels = []

    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = []
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        # Remove empty
        preb_label = [i for i in preb_label if i!=len(CHARS)-1]  
        label = "".join(map(lambda i: CHARS[i], preb_label))
        labels.append(label)

    return labels
 

def inference():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    img = cv2.imread(args.test_img)
    print(run(lprnet, args, img)[0])
    
if __name__=="__main__":
    inference()
