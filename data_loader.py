

import json
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda'):

        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        print('Project Directory:')
        print(project_directory)
        data_path = '/home/chenlb/datasets/DR/APTOS/w224n808'
        print('Data Path')
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        if self.dataset == 'CUB':
            self.datadir = self.data_path + '/CUB/'
        elif self.dataset == 'SUN':
            self.datadir = self.data_path + '/SUN/'
        elif self.dataset == 'AWA1':
            self.datadir = self.data_path + '/AWA1/'
        elif self.dataset == 'AWA2':
            self.datadir = self.data_path + '/AWA2/'
        elif self.dataset == 'APY':
            self.datadir = self.data_path + '/APY/'
        elif self.dataset == 'FLO':
            self.datadir = self.data_path + '/FLO/'
        elif self.dataset == "ZDFY" or self.dataset == "ADNI" or self.dataset == "APTOS":
            self.datadir = self.data_path

        if self.dataset == "ZDFY":
            self.read_turmor()
        elif self.dataset == "ADNI":
            self.read_adni()
        elif self.dataset == "APTOS":
            self.read_aptos()
        else:
            self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_aptos(self):
        self.train_class = [0, 2, 3, 4]

        # 加载训练特征和标签并转换为 Tensor
        self.train_feature = np.load(os.path.join(self.datadir, 'resnet101', 'train_features.npy'))  # (606, 2048)
        self.train_label = np.load(os.path.join(self.datadir, 'resnet101', 'train_targets.npy'))

        mask = np.isin(self.train_label, self.train_class)
        self.train_feature = self.train_feature[mask]
        self.train_label = self.train_label[mask]

        self.train_feature = torch.tensor(self.train_feature, dtype=torch.float32).to(self.device)
        self.train_label = torch.tensor(self.train_label, dtype=torch.long).to(self.device)

        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = 4
        self.ntest_class = 1
        
        
        # 加载测试特征和标签并转换为 Tensor
        self.test_feature = np.load(os.path.join(self.datadir, 'resnet101', 'valid_features.npy'))  # (171, 2048)
        self.test_label = np.load(os.path.join(self.datadir, 'resnet101', 'valid_targets.npy'))
        self.test_feature = torch.tensor(self.test_feature, dtype=torch.float32).to(self.device)
        self.test_label = torch.tensor(self.test_label, dtype=torch.long).to(self.device)

        # 加载属性嵌入并转换为 Tensor
        file_path = os.path.join(self.datadir, 'att', 'embeddings.json')
        with open(file_path, 'r') as f:
            data = json.load(f)

        attribute = {}
        for key, value in data.items():
            attribute[key] = np.array(value)

        categories = list(attribute.keys())
        embedding_list = [attribute[category] for category in categories]
        self.attribute = torch.tensor(np.array(embedding_list), dtype=torch.float32).to(self.device)
        self.aux_data = self.attribute

        self.allclasses = torch.from_numpy(np.array([0, 1, 2, 3, 4])).to(self.device)
        self.seenclasses = torch.from_numpy(np.array([0, 2, 3, 4])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.array([1])).to(self.device)
        self.attribute_seen = self.attribute[self.seenclasses]
        
        self.train_att = self.attribute_seen
        self.test_att = self.attribute[self.unseenclasses]

        # 提取标签为 1 和 2 的测试数据
        indices_seen = (self.test_label == 0) | (self.test_label == 2) | (self.test_label == 3) | (self.test_label == 4)
        self.test_seen_feature = self.test_feature[indices_seen]
        self.test_seen_label = self.test_label[indices_seen]

        # 提取标签为 0 的测试数据
        indices_unseen = self.test_label == 1
        self.test_unseen_feature = self.test_feature[indices_unseen]
        self.test_unseen_label = self.test_label[indices_unseen]

        # 计算每个类别的样本数量
        self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        
        # self.train_label = map_label(self.train_label, self.seenclasses)
        # self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        # self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        
        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = self.train_feature
        self.data['train_seen']['labels'] = self.train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[self.train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = self.test_seen_feature
        self.data['test_seen']['labels'] = self.test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = self.test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[self.test_unseen_label]
        self.data['test_unseen']['labels'] = self.test_unseen_label
        
        self.unseenclass_aux_data = self.aux_data[self.unseenclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

        # 打印结果以确认
        print("训练集特征形状:", self.train_feature.shape)
        print("训练集标签形状:", self.train_label.shape)
        print("测试集特征形状:", self.test_feature.shape)
        print("测试集标签形状:", self.test_label.shape)
        print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
        print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
        print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
        print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
        print("每个类别的样本数量:", self.train_samples_class_index)
    
    
    def read_adni(self):
        self.train_class = [0, 2]

        # 加载训练特征和标签并转换为 Tensor
        self.train_feature = np.load(os.path.join(self.datadir, 'resnet101', 'train_features.npy'))  # (606, 2048)
        self.train_label = np.load(os.path.join(self.datadir, 'resnet101', 'train_targets.npy'))
        self.train_feature = torch.tensor(self.train_feature, dtype=torch.float32).to(self.device)
        self.train_label = torch.tensor(self.train_label, dtype=torch.long).to(self.device)

        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = 2
        self.ntest_class = 1
        
        
        # 加载测试特征和标签并转换为 Tensor
        self.test_feature = np.load(os.path.join(self.datadir, 'resnet101', 'valid_features.npy'))  # (171, 2048)
        self.test_label = np.load(os.path.join(self.datadir, 'resnet101', 'valid_targets.npy'))
        self.test_feature = torch.tensor(self.test_feature, dtype=torch.float32).to(self.device)
        self.test_label = torch.tensor(self.test_label, dtype=torch.long).to(self.device)

        # 加载属性嵌入并转换为 Tensor
        file_path = os.path.join(self.datadir, 'att', 'embeddings.json')
        with open(file_path, 'r') as f:
            data = json.load(f)

        attribute = {}
        for key, value in data.items():
            attribute[key] = np.array(value)

        categories = list(attribute.keys())
        embedding_list = [attribute[category] for category in categories]
        self.attribute = torch.tensor(np.array(embedding_list), dtype=torch.float32).to(self.device)
        self.aux_data = self.attribute

        self.allclasses = torch.from_numpy(np.array([0, 1, 2])).to(self.device)
        self.seenclasses = torch.from_numpy(np.array([0, 2])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.array([1])).to(self.device)
        self.attribute_seen = self.attribute[self.seenclasses]
        
        self.train_att = self.attribute_seen
        self.test_att = self.attribute[self.unseenclasses]

        # 提取标签为 1 和 2 的测试数据
        indices_seen = (self.test_label == 0) | (self.test_label == 2)
        self.test_seen_feature = self.test_feature[indices_seen]
        self.test_seen_label = self.test_label[indices_seen]

        # 提取标签为 0 的测试数据
        indices_unseen = self.test_label == 1
        self.test_unseen_feature = self.test_feature[indices_unseen]
        self.test_unseen_label = self.test_label[indices_unseen]

        # 计算每个类别的样本数量
        self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        
        # self.train_label = map_label(self.train_label, self.seenclasses)
        # self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        # self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        
        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = self.train_feature
        self.data['train_seen']['labels'] = self.train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[self.train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = self.test_seen_feature
        self.data['test_seen']['labels'] = self.test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = self.test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[self.test_unseen_label]
        self.data['test_unseen']['labels'] = self.test_unseen_label
        
        self.unseenclass_aux_data = self.aux_data[self.unseenclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

        # 打印结果以确认
        print("训练集特征形状:", self.train_feature.shape)
        print("训练集标签形状:", self.train_label.shape)
        print("测试集特征形状:", self.test_feature.shape)
        print("测试集标签形状:", self.test_label.shape)
        print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
        print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
        print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
        print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
        print("每个类别的样本数量:", self.train_samples_class_index)


    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label =  self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        
        return [ batch_feature, batch_att, batch_label]
        
    def next_unseen_batch(self, batch_size):
        idx_unseen = torch.randperm(self.ntest_unseen)[0:batch_size]
        # batch_unseen_feature = self.data['test_unseen']['resnet_features'][idx_unseen]
        batch_unseen_label =  self.data['test_unseen']['labels'][idx_unseen]
        batch_unseen_att = self.aux_data[batch_unseen_label]
        return [batch_unseen_att, batch_unseen_label]
        
    # def next_unseen_batch(self, batch_size):
        # idx_unseen = torch.randperm(self.ntest_unseen)[0:batch_size]
        # batch_unseen_feature = self.data['test_unseen']['resnet_features'][idx_unseen]
        
        # idx_unseen =torch.randint(0, self.unseenclasses.shape[0], (batch_size,))
        # unseen_label=self.unseenclasses[idx_unseen]
        # batch_unseen_att=self.aux_data[unseen_label]
        
        # return [batch_unseen_att, unseen_label]
        
    def read_turmor(self):
        
        self.train_class = [1, 2]

        # 加载训练特征和标签并转换为 Tensor
        self.train_feature = np.load(os.path.join(self.datadir, 'resnet101', 'train_features.npy'))  # (606, 2048)
        self.train_label = np.load(os.path.join(self.datadir, 'resnet101', 'train_targets.npy'))
        self.train_feature = torch.tensor(self.train_feature, dtype=torch.float32).to(self.device)
        self.train_label = torch.tensor(self.train_label, dtype=torch.long).to(self.device)

        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = 2
        self.ntest_class = 1
        
        
        # 加载测试特征和标签并转换为 Tensor
        self.test_feature = np.load(os.path.join(self.datadir, 'resnet101', 'valid_features.npy'))  # (171, 2048)
        self.test_label = np.load(os.path.join(self.datadir, 'resnet101', 'valid_targets.npy'))
        self.test_feature = torch.tensor(self.test_feature, dtype=torch.float32).to(self.device)
        self.test_label = torch.tensor(self.test_label, dtype=torch.long).to(self.device)

        # 加载属性嵌入并转换为 Tensor
        file_path = os.path.join(self.datadir, 'att', 'embeddings.json')
        with open(file_path, 'r') as f:
            data = json.load(f)

        attribute = {}
        for key, value in data.items():
            attribute[key] = np.array(value)

        categories = list(attribute.keys())
        embedding_list = [attribute[category] for category in categories]
        self.attribute = torch.tensor(np.array(embedding_list), dtype=torch.float32).to(self.device)
        self.aux_data = self.attribute

        self.allclasses = torch.from_numpy(np.array([0, 1, 2])).to(self.device)
        self.seenclasses = torch.from_numpy(np.array([1, 2])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.array([0])).to(self.device)
        self.attribute_seen = self.attribute[self.seenclasses]
        
        self.train_att = self.attribute_seen
        self.test_att = self.attribute[self.unseenclasses]

        # 提取标签为 1 和 2 的测试数据
        indices_seen = (self.test_label == 1) | (self.test_label == 2)
        self.test_seen_feature = self.test_feature[indices_seen]
        self.test_seen_label = self.test_label[indices_seen]

        # 提取标签为 0 的测试数据
        indices_unseen = self.test_label == 0
        self.test_unseen_feature = self.test_feature[indices_unseen]
        self.test_unseen_label = self.test_label[indices_unseen]

        # 计算每个类别的样本数量
        self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        
        # self.train_label = map_label(self.train_label, self.seenclasses)
        # self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        # self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        
        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = self.train_feature
        self.data['train_seen']['labels'] = self.train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[self.train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = self.test_seen_feature
        self.data['test_seen']['labels'] = self.test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = self.test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[self.test_unseen_label]
        self.data['test_unseen']['labels'] = self.test_unseen_label
        
        self.unseenclass_aux_data = self.aux_data[self.unseenclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

        # 打印结果以确认
        print("训练集特征形状:", self.train_feature.shape)
        print("训练集标签形状:", self.train_label.shape)
        print("测试集特征形状:", self.test_feature.shape)
        print("测试集标签形状:", self.test_label.shape)
        print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
        print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
        print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
        print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
        print("每个类别的样本数量:", self.train_samples_class_index)

    def read_matdataset(self):

        path= self.datadir + 'res101.mat'
        print('_____')
        print(path)
        matcontent = sio.loadmat(path)
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1

        path= self.datadir + 'att_splits.mat'
        matcontent = sio.loadmat(path)
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1


        if self.auxiliary_data_source == 'attributes':
            self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary datasource is not available for this dataset')
            else:

                with open(self.datadir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = torch.from_numpy(x[self.auxiliary_data_source]).float().to(self.device)


                print('loaded ', self.auxiliary_data_source)


        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.transform(feature[test_seen_loc])
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)
        self.ntest_unseen = test_unseen_feature.size()[0]
        
        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.unseenclass_aux_data = self.aux_data[self.unseenclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]


    def transfer_features(self, n, num_queries='num_features'):
        print('size before')
        print(self.data['test_unseen']['resnet_features'].size())
        print(self.data['train_seen']['resnet_features'].size())


        print('o'*100)
        print(self.data['test_unseen'].keys())
        for i,s in enumerate(self.unseenclasses):

            features_of_that_class   = self.data['test_unseen']['resnet_features'][self.data['test_unseen']['labels']==s ,:]

            if 'attributes' == self.auxiliary_data_source:
                attributes_of_that_class = self.data['test_unseen']['attributes'][self.data['test_unseen']['labels']==s ,:]
                use_att = True
            else:
                use_att = False
            if 'sentences' == self.auxiliary_data_source:
                sentences_of_that_class = self.data['test_unseen']['sentences'][self.data['test_unseen']['labels']==s ,:]
                use_stc = True
            else:
                use_stc = False
            if 'word2vec' == self.auxiliary_data_source:
                word2vec_of_that_class = self.data['test_unseen']['word2vec'][self.data['test_unseen']['labels']==s ,:]
                use_w2v = True
            else:
                use_w2v = False
            if 'glove' == self.auxiliary_data_source:
                glove_of_that_class = self.data['test_unseen']['glove'][self.data['test_unseen']['labels']==s ,:]
                use_glo = True
            else:
                use_glo = False
            if 'wordnet' == self.auxiliary_data_source:
                wordnet_of_that_class = self.data['test_unseen']['wordnet'][self.data['test_unseen']['labels']==s ,:]
                use_hie = True
            else:
                use_hie = False


            num_features = features_of_that_class.size(0)

            indices = torch.randperm(num_features)

            if num_queries!='num_features':

                indices = indices[:n+num_queries]


            print(features_of_that_class.size())


            if i==0:

                new_train_unseen      = features_of_that_class[   indices[:n] ,:]

                if use_att:
                    new_train_unseen_att  = attributes_of_that_class[ indices[:n] ,:]
                if use_stc:
                    new_train_unseen_stc  = sentences_of_that_class[ indices[:n] ,:]
                if use_w2v:
                    new_train_unseen_w2v  = word2vec_of_that_class[ indices[:n] ,:]
                if use_glo:
                    new_train_unseen_glo  = glove_of_that_class[ indices[:n] ,:]
                if use_hie:
                    new_train_unseen_hie  = wordnet_of_that_class[ indices[:n] ,:]


                new_train_unseen_label  = s.repeat(n)

                new_test_unseen = features_of_that_class[  indices[n:] ,:]

                new_test_unseen_label = s.repeat( len(indices[n:] ))

            else:
                new_train_unseen  = torch.cat(( new_train_unseen             , features_of_that_class[  indices[:n] ,:]),dim=0)
                new_train_unseen_label  = torch.cat(( new_train_unseen_label , s.repeat(n)),dim=0)

                new_test_unseen =  torch.cat(( new_test_unseen,    features_of_that_class[  indices[n:] ,:]),dim=0)
                new_test_unseen_label = torch.cat(( new_test_unseen_label  ,s.repeat( len(indices[n:]) )) ,dim=0)

                if use_att:
                    new_train_unseen_att    = torch.cat(( new_train_unseen_att   , attributes_of_that_class[indices[:n] ,:]),dim=0)
                if use_stc:
                    new_train_unseen_stc    = torch.cat(( new_train_unseen_stc   , sentences_of_that_class[indices[:n] ,:]),dim=0)
                if use_w2v:
                    new_train_unseen_w2v    = torch.cat(( new_train_unseen_w2v   , word2vec_of_that_class[indices[:n] ,:]),dim=0)
                if use_glo:
                    new_train_unseen_glo    = torch.cat(( new_train_unseen_glo   , glove_of_that_class[indices[:n] ,:]),dim=0)
                if use_hie:
                    new_train_unseen_hie    = torch.cat(( new_train_unseen_hie   , wordnet_of_that_class[indices[:n] ,:]),dim=0)



        print('new_test_unseen.size(): ', new_test_unseen.size())
        print('new_test_unseen_label.size(): ', new_test_unseen_label.size())
        print('new_train_unseen.size(): ', new_train_unseen.size())
        #print('new_train_unseen_att.size(): ', new_train_unseen_att.size())
        print('new_train_unseen_label.size(): ', new_train_unseen_label.size())
        print('>> num unseen classes: ' + str(len(self.unseenclasses)))

        #######
        ##
        #######

        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)
        #self.data['train_seen']['resnet_features']  = copy.deepcopy(new_train_seen)

        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)
        #self.data['train_seen']['labels']  = copy.deepcopy(new_train_seen_label)

        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data['train_unseen']['resnet_features'].size(0)

        if use_att:
            self.data['train_unseen']['attributes'] = copy.deepcopy(new_train_unseen_att)
        if use_w2v:
            self.data['train_unseen']['word2vec']   = copy.deepcopy(new_train_unseen_w2v)
        if use_stc:
            self.data['train_unseen']['sentences']  = copy.deepcopy(new_train_unseen_stc)
        if use_glo:
            self.data['train_unseen']['glove']      = copy.deepcopy(new_train_unseen_glo)
        if use_hie:
            self.data['train_unseen']['wordnet']   = copy.deepcopy(new_train_unseen_hie)

        ####
        self.data['train_seen_unseen_mixed'] = {}
        self.data['train_seen_unseen_mixed']['resnet_features'] = torch.cat((self.data['train_seen']['resnet_features'],self.data['train_unseen']['resnet_features']),dim=0)
        self.data['train_seen_unseen_mixed']['labels'] = torch.cat((self.data['train_seen']['labels'],self.data['train_unseen']['labels']),dim=0)

        self.ntrain_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)

        if use_att:
            self.data['train_seen_unseen_mixed']['attributes'] = torch.cat((self.data['train_seen']['attributes'],self.data['train_unseen']['attributes']),dim=0)
        if use_w2v:
            self.data['train_seen_unseen_mixed']['word2vec'] = torch.cat((self.data['train_seen']['word2vec'],self.data['train_unseen']['word2vec']),dim=0)
        if use_stc:
            self.data['train_seen_unseen_mixed']['sentences'] = torch.cat((self.data['train_seen']['sentences'],self.data['train_unseen']['sentences']),dim=0)
        if use_glo:
            self.data['train_seen_unseen_mixed']['glove'] = torch.cat((self.data['train_seen']['glove'],self.data['train_unseen']['glove']),dim=0)
        if use_hie:
            self.data['train_seen_unseen_mixed']['wordnet'] = torch.cat((self.data['train_seen']['wordnet'],self.data['train_unseen']['wordnet']),dim=0)

#d = DATA_LOADER()
