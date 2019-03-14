from __future__ import print_function
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.embedding = nn.Linear(512, nclass)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.embedding(conv)


        return output

def val(net, test_image):
    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    preds = crnn(test_image.float()) #T*1*C
    probs = torch.nn.Softmax(dim=2)(preds) #T*1*C
    probs = probs.squeeze()
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1) #T
    string, score = decode(preds.data,probs)
    return string, score

def decode(t,probs):
    char_list = []
    score = 1.
    for i in range(t.size(0)):
       if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
           char_prob = probs[i,t[i]]
           if score > char_prob:
               score = char_prob
           char_list.append(alphabet[t[i] - 1])
    final_string = ''.join(char_list)
    return final_string,score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='image path')
    parser.add_argument('model', type=str, help="path to model")
    parser.add_argument('--threshold', type=float, default=0.6, help='char thresh')

    opt = parser.parse_args()
    print(opt)


    cudnn.benchmark = True


    alphabet =  u'皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'
    
    img = cv2.imread(opt.image_path,0)
    img = img.astype(float)
    img = cv2.resize(img, (100, 32))
    img = np.expand_dims(img, axis=2)
    img /= 255
    img -= 0.5
    img /= 0.5

    img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0)
    nclass = len(alphabet) + 1

    crnn = CRNN(32, 1, nclass, 256)
    pretrained_dict = torch.load(opt.model)
    pretrained_load_dict = dict()
    for k,v in pretrained_dict.items():
        new_k = k.replace('module.','')
        pretrained_load_dict[new_k]=v
    crnn.load_state_dict(pretrained_load_dict)

    string,score = val(crnn, img_tensor)
    print('\nimage_path:{},\nrecognition:{},\nconfidence:{},\n>threshold:{}'.format(opt.image_path,string,score,score>opt.threshold))
