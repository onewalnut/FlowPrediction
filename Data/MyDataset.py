# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyDataset
   Description :
   Author :       walnut
   date:          2021/1/5
-------------------------------------------------
   Change Activity:
                  2021/1/5:
-------------------------------------------------
"""
__author__ = 'walnut'

from classes import *
from paras import *
from Utils.Reader import read_from_json


def default_transform(data):
    return torch.from_numpy(data.astype(np.float))


class SeqDataset(Dataset):
    def __init__(self, train, file, transformer=default_transform, target_transformer=default_transform):
        raw_data = read_from_json(file)
        if len(np.array(raw_data).shape) < 3:
            raw_data = [raw_data]

        data = self.data_process(raw_data)
        seqs, labels = self.calibrate(data, shuffle=True)

        cut_point = int(len(seqs)*0.7)
        if train:
            self.seqs = seqs[0:cut_point]
            self.labels = labels[0:cut_point]
        else:
            self.seqs = seqs[cut_point:]
            self.labels = labels[cut_point:]

        # self.seqs = (seqs - np.min(seqs)) / (np.max(seqs) - np.min(seqs))
        self.transformer = transformer
        self.target_transformer = target_transformer


    def __getitem__(self, index):
        my_input = self.seqs[index]
        label = self.labels[index]

        # if self.transformer is not None:
        #     seq = self.transformer(seq)
        # if self.target_transformer is not None:
        #     label = self.target_transformer(label)

        return my_input, label


    def __len__(self):
        return len(self.seqs)


    def data_process(self, data):
        data_procesed = []
        for seq in data:
            seq = np.array(seq)
            hour = np.array([date[-2:] for date in seq[:, 0]])
            seq = np.concatenate((np.expand_dims(hour, axis=1), seq[:, 1:3]), axis=1)
            data_procesed.append(np.array(seq, dtype=np.float))
        return data_procesed


    def calibrate(self, data, shuffle=False):
        seqs = []
        labels = []
        for seq in data:
            # if len(seq) < 300:
            #     continue

            # shuffle config
            seqs_tmp = []
            for idx in range(0, len(seq)-(SEQ_LEN+1), CALIBRATE_INTERVAL):
                # if np.average(seq[idx: idx + SEQ_LEN + 1, 1]) < 1000:
                #     continue
                seqs_tmp.append(seq[idx: idx+SEQ_LEN+1])
            if shuffle:
                random.shuffle(seqs_tmp)

            for item in seqs_tmp:
                # input size
                tmp = item[:, 1]
                tmp = np.array(tmp)

                seqs.append(np.expand_dims(tmp[0:SEQ_LEN], axis=1))
                labels.append([tmp[SEQ_LEN]])


            # normal config
            # for idx in range(0, len(seq)-(SEQ_LEN+1), CALIBRATE_INTERVAL):
            #
            #     tmp = seq[idx: idx + SEQ_LEN + 1, 1]
            #     tmp = np.array(tmp)
            #     # tmp = (np.array(tmp) - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
            #
            #     seqs.append(np.expand_dims(tmp[0:SEQ_LEN], axis=1))
            #     labels.append([tmp[SEQ_LEN]])

                # seqs.append(seq[idx: idx + SEQ_LEN])
                # labels.append(seq[idx + SEQ_LEN])

        return np.array(seqs), np.array(labels)
