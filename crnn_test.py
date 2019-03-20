from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
import models.keys as keys
import sys
reload(sys)
sys.setdefaultencoding('utf8')


parser = argparse.ArgumentParser()
parser.add_argument('--valroot', type=str, default=None, help='val text data list')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')

opt = parser.parse_args()
print(opt)


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


val_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))
assert val_dataset

nclass = len(keys.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(keys.alphabet, ignore_case=True)
criterion = CTCLoss()

crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn = torch.nn.DataParallel(crnn).cuda()
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    pretrained_dict = torch.load(opt.crnn)
    pretrained_load_dict = dict()
    for k,v in pretrained_dict.items():
        new_k = k.replace('module.','')
        pretrained_load_dict[new_k]=v

    crnn.load_state_dict(pretrained_load_dict)

print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

image = image.cuda()
criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    n_count = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image) # Time*Batchsize*Class
        preds = preds.permute(1, 0, 2)
        probs = F.softmax(preds,dim=2) # Time*Batchsize*Class
        probs = probs.transpose(1,0).contiguous()
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2) # Time*Batchsize*1
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1) # Batchsize*Time
        sim_preds = converter.decode_test(preds.data, preds_size.data, probs=probs, raw=False, threshold=0.6)
        for pred, target in zip(sim_preds, cpu_texts):
            if len(pred)>=10:
                continue
            n_count += 1
            if pred == target.lower():
                n_correct += 1

        #raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
        for pred, gt in zip(sim_preds, cpu_texts):
            print(pred)
            print(gt)
            print('-'*20)
            #print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = float(n_correct) / n_count

    print('Test loss: %f, accuray: %f(%d/%d)' % (loss_avg.val(), accuracy,n_correct,n_count))



val(crnn, val_dataset, criterion, 100000000)
