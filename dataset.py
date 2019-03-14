#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import models.keys as keys

DEBUG = False

class textDataset(Dataset):
    def __init__(self, datalist, fonts, font_sizes="35-42", transform=None):
        '''Initialization of textDataset
        args
        bg_path: (string) background image path, if is None, use white background
        fonts: (string) font file path, multi fonts is seperated by ','
        font_sizes: (string) text font size range, "min_size - max_size"
        '''
        font_sizes = font_sizes.split('-')
        self.font_sizes = range(int(font_sizes[0]), int(font_sizes[1]))
        self.fonts = fonts.split(',')
        self.alphabet = keys.alphabet
        self.transform = transform
        text_samples = []
        with open(datalist,'r') as f:
            for line in f:
                text_samples.append(line.strip())
        self.text_samples = text_samples

        self.nSamples = len(self.text_samples)
        self.punctuation = [',','。','<','>','@',':','“','”','"','\'','~',
                            '《','》','[',']','#','=','±','+','_','-','°',
                            '!','！',';','.','/','%','*','(',')','?','？',
                            '、','￡','￥','$','〈','〉','{','}','//','\\']
        self.digits = ['0','1','2','3','4','5','6','7','8','9']


    def __len__(self):
        return self.nSamples


    def _generate_sample(self, index):
        txt1 = self.text_samples[index]
        while txt1 == "":
            print("txt is empty,random get one")
            txt1 = random.choice(self.text_samples)
        choice = random.randint(1, 8)
        # 5 choices:
        # 1-4: word
        # 5: word + p(punctuation)
        # 6: p + word
        # 7: word1 + p + word2
        # 8: digit + p + word
        if choice <=4:
            sample = txt1
        elif choice == 5:
            p = random.choice(self.punctuation)
            sample = txt1 + p
        elif choice == 6:
            p = random.choice(self.punctuation)
            sample = p + txt1
        elif choice == 7:
            p = random.choice(self.punctuation)
            txt2 = random.choice(self.text_samples)
            sample = txt1 + p + txt2
        else:
            p = random.choice(self.punctuation)
            digit = random.choice(self.digits)
            sample = digit + p + txt1

        # If the char is not in the keys, randomly replace it
        textdata = ""
        for o_char in sample.decode('utf-8'):
            if self.alphabet.find(o_char) == -1:
                o_char = random.choice(self.alphabet) #self.alphabet[random.randint(0, len(self.alphabet)-1)]
            textdata += o_char

        return textdata


    def __getitem__(self, index):
        index = index % self.nSamples

        text = self._generate_sample(index)
        # initialize font
        font = random.choice(self.fonts)
        font_size = random.choice(self.font_sizes)
        font = ImageFont.truetype(font,font_size)

        text_w, text_h = font.getsize(text)
        bg = Image.new("RGB", (text_w, text_h), "white")
        image = ImageDraw.Draw(bg)
        image.text((0, 0), text, font=font, fill='black')
        bg = bg.convert('L')

        if DEBUG:
            bg.save(os.path.join('data/debug/', str(index) + '_' + text + ".JPEG"))

        if self.transform is not None:
            bg = self.transform(bg)

        return (bg, ''.join(text))

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
