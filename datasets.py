from utils import *
import os, numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, SVHN
from torchvision import transforms
from PIL import Image


DS_INFO = {
    'mnist': {'norm': [(0.5,), (0.5,)],
            'n_classes': 10,
            'shape': (1, 28, 28)},
    'fashion': {'norm': [(0.5,), (0.5,)],
            'n_classes': 10,
            'shape': (1, 28, 28)},
    'svhn': {'norm': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
            'n_classes': 10,
            'shape': (3, 32, 32)},
    'cifar10': {'norm': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
            'n_classes': 10,
            'shape': (3, 32, 32)},
    'cifar100': {'norm': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
            'n_classes': 100,
            'shape': (3, 32, 32)},
    'tiny': {'norm': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.200)],
            'n_classes': 200,
            'shape': (3, 32, 32)}
}

BMRK = {'mnist':MNIST, 'fashion':FashionMNIST, 'svhn':SVHN, 'cifar10':CIFAR10, 'cifar100':CIFAR100}

def get_transform(dname, aug=False):
    if aug:
        return transforms.Compose([
                    transforms.RandomResizedCrop(DS_INFO[dname]['shape'][-1]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(*DS_INFO[dname]['norm'])])
    else:
        return transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize(*DS_INFO[dname]['norm'])])


class TruncSubset(Dataset):
    def __init__(self, ds, indexs):
        super().__init__()
        self.ds = ds
        self.indexs = indexs
    
    def __getitem__(self, index):
        return self.ds[self.indexs[index]]
    
    def __len__(self):
        return len(self.indexs)

class FlxSet(Dataset):
    def __init__(self, dname, data=None, targets=None, norm_ds=None, aug=False, indexs=None):
        super().__init__()
        self.dname = dname
        self.data = data
        self.targets = targets
        self.norm_ds = norm_ds
        self.indexs = indexs
        self.update_transform(aug)
    
    def update_transform(self, aug):
        self.transform = get_transform(self.dname, True) if aug else None

    def __getitem__(self, index):
        idx = index if self.indexs is None else self.indexs[index]
        if self.transform is None: return self.norm_ds[idx]
        
        img = self.data[idx]
        if self.dname == 'svhn':
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        elif self.dname in ['mnist', 'fashion']:
            img = Image.fromarray(img.numpy(), mode="L")
        elif self.dname in ['cifar10', 'cifar100']:
            img = Image.fromarray(img)

        img = self.transform(img)
        return img, self.targets[idx]
    
    def __len__(self):
        return len(self.data) if self.indexs is None else len(self.indexs)
    
    def pick(self, indexs):
        return FlxSet(self.dname, self.data, self.targets, 
                    self.norm_ds, self.transform is not None, indexs)
    
    def enrich2num(self, num):
        n_total = self.__len__()
        n_base = num // n_total
        indexs = list(range(n_total)) * n_base
        indexs.extend(list(range(n_total))[:num % n_total])
        return self.pick(indexs)

class BMSet(Dataset):
    def __init__(self, dname, data_dir='brenchmarks', is_train=True):
        super().__init__()
        self.dname = dname
        self.ds = self.norm_ds(dname, data_dir, is_train)
    
    def norm_ds(self, dname, data_dir, is_train):
        root = os.path.join(data_dir, dname)
        if dname == 'svhn': train = 'train' if is_train else 'test'
        else: train = is_train
        transform = get_transform(dname)
        return BMRK[dname](root, train, transform, None, download=True)
    
    def __getitem__(self, index):
        return self.ds[index]
    
    def __len__(self):
        return len(self.ds)

    def query_targets(self, indexs=None):
        targets = self.ds.labels if self.dname=='svhn' else self.ds.targets
        res = np.array(deepcopy(targets))
        if indexs is not None: res = res[indexs]
        return res
    
    def unit_sample(self, label_indexs, num):
        counts_cur = np.array([len(e) for e in label_indexs])
        n_total_cur = sum(counts_cur)
        npc = np.array([num / n_total_cur] * len(label_indexs)) * counts_cur
        npc = npc.astype(int)
        # print(f'num={num}, sum(npc)={sum(npc)}')
        npc[:num - sum(npc)] += 1
        indexs = []
        pop_list = []
        for i in range(len(label_indexs)):
            idxs_chosen = np.random.choice(label_indexs[i], npc[i], replace=False)
            indexs.extend(list(idxs_chosen))
            label_indexs[i] = list(set(label_indexs[i]) - set(idxs_chosen))
            if len(label_indexs[i]) == 0: pop_list.append(i)

        for j in range(len(pop_list)-1, -1, -1): label_indexs.pop(pop_list[j])

        return indexs

    def get_subset(self, indexs, aug=False, one_way=True):
        if not aug and one_way:
            return TruncSubset(self.ds, indexs)
        
        data, targets = None, None
        data = self.ds.data
        targets = self.ds.labels if self.dname=='svhn' else self.ds.targets
        data = data[indexs]
        targets = np.array(targets)[indexs]
        norm_ds = None if aug else TruncSubset(self.ds, indexs)
        return FlxSet(self.dname, data, targets, norm_ds, aug)
    
    def unit_partition(self, index_list):
        assert sum(index_list) == self.__len__()
        targets = self.query_targets()
        labels = np.unique(targets)
        label_indexs = []
        for c in labels:
            idxs = np.where(targets == c)[0]
            label_indexs.append(list(idxs))
        
        sub_index_list = []
        for i in range(len(index_list)-1):
            num = index_list[i]
            if num <= 0:
                sub_index_list.append(None)
                continue

            indexs = self.unit_sample(label_indexs, num)
            sub_index_list.append(np.array(indexs))
        
        indexs = []
        for e in label_indexs:
            indexs.extend(e)
        sub_index_list.append(np.array(indexs))
        return sub_index_list

def insert_batch(obj, batch_data, gcal=False):
    if batch_data is None: return obj
    elif obj is None:
        if gcal: return batch_data
        else: return batch_data.cpu().detach()
    else:
        if gcal: return torch.cat([obj, batch_data], dim=0)
        else: return torch.cat([obj, batch_data.cpu().detach()], dim=0)

def obj_slice(obj, indexs):
    if obj is None: return None
    else: return obj[indexs]

class BasicDataset(Dataset):
    def __init__(self, n_fields=7, fields=None):
        super().__init__()
        if fields is None:
            self.n_fields = n_fields
            self.fields = [None] * n_fields
        else:
            self.fields = fields
            self.n_fields = len(self.fields)

        # self.n_fields = n_fields
        # self.fields = [None] * n_fields if fields is None else fields
        
    def __getitem__(self, index):
        results = []
        for i in range(self.n_fields):
            item = self.fields[i]
            if item is not None:
                results.append(item[index])

        return results[0] if len(results) == 1 else tuple(results)
        
    def __len__(self):
        num = 0
        for i in range(self.n_fields):
            item = self.fields[i]
            if item is not None:
                num = len(item)
                break
        return num
    
    def query_num(self, i_field, value):
        item = self.fields[i_field]
        if item is None: return 0
        else: return (item == value).sum().item()
    
    def insert(self, fields):
        for i in range(len(fields)):
            self.fields[i] = insert_batch(self.fields[i], fields[i])
    
    def extend(self, ds):
        self.insert(ds.fields)


def sp_get_publ(priv, publ, data_dir='brenchmarks', is_train=True):
    root = os.path.join(data_dir, publ)
    if publ == 'svhn': train = 'train' if is_train else 'test'
    else: train = is_train

    if 'fashion' in priv and 'cifar' in publ:
        transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((28,28)),
                        transforms.ToTensor(),
                        transforms.Normalize(*DS_INFO[priv]['norm'])])
    elif 'cifar' in priv and 'fashion' in publ:
        transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.Resize((32,32)),
                        transforms.ToTensor(),
                        transforms.Normalize(*DS_INFO[priv]['norm'])])
    else:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(*DS_INFO[priv]['norm'])])
        
    ds = BMRK[publ](root, train, transform, None, download=True)
    
    dloader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    inputs, targets = None, None
    for x, y in dloader:
        inputs, targets = x, y

    return BasicDataset(fields=[inputs, targets])


if __name__ == '__main__':
    # dname = 'cifar10'
    # root = os.path.join('brenchmarks', dname)
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((28,28)),
    #     transforms.ToTensor(),
    #                 transforms.Normalize(*DS_INFO['mnist']['norm'])])
    # ds = BMRK[dname](root, True, transform, None, download=False)

    ds = sp_get_publ(priv='cifar10', publ='mnist', data_dir='brenchmarks', is_train=False)
    dloader = DataLoader(ds, batch_size=4)
    for x, y in dloader:
        print(x.shape)
        print(y.shape)
        break