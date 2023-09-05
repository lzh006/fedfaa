import os, time, logging
import torch, random, numpy as np
from copy import deepcopy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import BMSet, sp_get_publ
from models import *

DATA_DIR = 'brenchmarks'
RES_DIR = 'results'

def create_log_dir(algo:str, timestamp=None):
    if timestamp is None:
        lt = time.localtime()
        timestamp = time.strftime("%y%m%d[%H%M%S]", lt)

    dir_paths = [DATA_DIR, RES_DIR]
    for d in dir_paths:
        if not os.path.exists(d): os.mkdir(d)

    root_path = os.path.join(RES_DIR, algo, timestamp)
    if not os.path.exists(root_path): os.makedirs(root_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(root_path, f'expr_{timestamp}.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S', level=logging.DEBUG, filemode='a+')
    logging.getLogger('nmslib').setLevel(logging.WARNING)
    
    return root_path, timestamp

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

def partition_data(y_train, n_clients, alpha):
    n_classes = len(set(y_train))
    n_gtrain = y_train.shape[0]

    idxs = np.random.permutation(n_gtrain)
    batch_idxs = np.array_split(idxs, n_clients)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}
    min_size = 0
    min_require_size = 10

    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    
    net_cls_counts = data_stats(y_train, net_dataidx_map)
    return net_dataidx_map, net_cls_counts

def log_label_dist(cls_counts, n_classes, fpath):
    n_clients = len(cls_counts)
    dist = np.zeros(shape=(n_clients, n_classes), dtype=int)
    cidxs = sorted(cls_counts.keys())
    for i in range(len(cidxs)):
        for label in cls_counts[cidxs[i]]:
            dist[i][label] = cls_counts[cidxs[i]][label]
    
    sum_per_client = dist.sum(axis=1)
    sum_per_label = dist.sum(axis=0)
    terms = ['cidx']
    terms.extend([str(label) for label in range(n_classes)])
    terms.append('sum')
    line = ','.join(terms) + '\n'
    for i in range(n_clients):
        terms = [str(cidxs[i])]
        terms.extend([str(e) for e in dist[i]])
        terms.append(str(sum_per_client[i]))
        line += ','.join(terms) + '\n'
    
    terms = ['sum']
    terms.extend([str(n) for n in sum_per_label])
    terms.append(str(sum(sum_per_client)))
    line += ','.join(terms) + '\n'
    with open(fpath, 'a+') as f:
        f.write(line)
        f.flush()

def load_fed_data(args, data_dir='brenchmarks', root_path=None):
    raw_ds = BMSet(args.dataset, data_dir)

    n_grdd = 0 if args.n_grdd is None or args.n_grdd <= 0 else args.n_grdd
    n_gp = 0 if args.n_gp is None or args.n_gp <= 0 else args.n_gp
    n_gtrain = len(raw_ds) - n_grdd - n_gp
    indexs_grdd, indexs_gp, indexs_gtrain = raw_ds.unit_partition([n_grdd, n_gp, n_gtrain])

    net_dataidx_map, net_cls_counts = partition_data(raw_ds.query_targets(indexs_gtrain), args.n_clients, args.alpha)
    if root_path is not None:
        log_label_dist(net_cls_counts, n_classes=len(set(raw_ds.query_targets())), 
            fpath=os.path.join(root_path, 
                    f'T-{args.dataset}-{args.alpha}-{args.n_clients}-{n_gtrain}-{args.seed}.csv'))
    
    cpriv_sets = []
    for cidx in range(args.n_clients):
        map_indexs = net_dataidx_map[cidx]
        indexs_client = indexs_gtrain[map_indexs]
        subset = raw_ds.get_subset(indexs_client, aug=args.aug_ltrain, one_way=('fedkfd' not in args.algo))
        cpriv_sets.append(subset)
    
    if 'dn_publ' in args and args.dn_publ is not None:
        # gp_ds = BMSet(args.dn_publ, data_dir)
        gp_ds = sp_get_publ(priv=args.dataset, publ=args.dn_publ, data_dir=data_dir, is_train=True)
    else:
        gp_ds = raw_ds.get_subset(indexs_gp) if n_gp > 0 else None
        
    gtest_ds = BMSet(args.dataset, data_dir, is_train=False)
    return cpriv_sets, gtest_ds, gp_ds
    
def create_model(arch, n_classes, img_shape, feats_loc=-1):
    if 'resnet' in arch:
        gnnp = 2 if '-GN' in arch else None
        size = int(arch[6:-3]) if '-GN' in arch else int(arch[6:])
        model = ResNet(nc=img_shape[0], in_dim=img_shape[-1], size=size, n_classes=n_classes, gnnp=gnnp, feats_loc=feats_loc)
    elif 'lenet' in arch:
        model = LeNet(nc=img_shape[0], in_dim=img_shape[-1], n_classes=n_classes, feats_loc=feats_loc)
    else:
        exit(f'Error: unrecognized model {arch}')

    return model

def create_gen(arch, nz, img_shape, ngf=8, n_classes=None):
    if arch == 'fast': obj = FastGen
    elif arch == 'dc': obj = DCGen
    return obj(nz=nz, img_shape=img_shape, n_classes=n_classes, ngf=ngf)
    
def client_sampling(i_iter, n_clients, n_active):
    if n_clients == n_active: return list(range(n_clients))
    np.random.seed(i_iter)
    return np.random.choice(range(n_clients), n_active, replace=False)


class EarlyStopping:
    def __init__(self, patience, reverse=False, count_mode=False, eq=False):
        self.patience = patience
        self.reverse = reverse
        self.count_mode = count_mode
        self.count = 0
        self.best_metrics = 1e20 if reverse else -1e20
        self.cache = None
        self.best_iter = 0
        self.eq = eq
    
    def cmp(self, v1, v2, lt, eq=False):
        if lt and eq: return v1 <= v2
        elif lt and not eq: return v1 < v2
        elif not lt and eq: return v1 >= v2
        else: return v1 > v2
    
    def watch(self, metrics, model, i_iter):
        is_better = self.cmp(metrics, self.best_metrics, 
                            lt=self.reverse, eq=self.eq)
        if is_better:
            self.count = 0
            self.best_metrics = metrics
            self.cache = deepcopy(model)
            self.best_iter = i_iter
        else:
            self.count += 1
        
        return self.count >= self.patience if self.count_mode else i_iter - self.best_iter >= self.patience
        
    def get_best(self):
        return self.cache, self.best_iter

class MetricsLogger:
    def __init__(self, fpath, title=None):
        self.fpath = fpath
        self.print_cache = f'{title}\n' if title is not None else ''
    
    def log(self, line):
        self.print_cache += f'{line}\n'
    
    def print(self):
        with open(self.fpath, 'a+') as f:
            f.write(self.print_cache)
            f.flush()
            self.print_cache = ''

class BestTracker:
    def __init__(self, reverse=False):
        self.reverse = reverse
        self.best_metrics = 1e20 if reverse else -1e20
        self.best_iter = 0
        
    def watch(self, metrics, i_iter):
        is_better = metrics < self.best_metrics if self.reverse else metrics > self.best_metrics
        if is_better:
            self.best_metrics = metrics
            self.best_iter = i_iter
    
    def get_best(self):
        return self.best_metrics, self.best_iter

def create_optim(params2optim, lr, mode, ft=True):
    po = filter(lambda p: p.requires_grad, params2optim) if ft else  params2optim
    if mode == 'adam': return Adam(po, lr)
    elif mode == 'sgd': return SGD(po, lr)
    else: exit(f'no such optim {mode}')

class MySched:
    def __init__(self, optimizer, epoch, mode, eta_min=2e-4):
        self.scheduler = None if mode is None else self.get_sched(optimizer, epoch, mode, eta_min)

    def get_sched(self, optimizer, epoch, mode, eta_min):
        if mode == 'cos':
            scheduler = CosineAnnealingLR(optimizer, epoch, eta_min=eta_min)
        elif mode == 'multi':
            scheduler = MultiStepLR(optimizer, milestones=[round(epoch*0.5), round(epoch*0.75)], gamma=0.5)
        else:
            exit(f'no such scheduler {mode}')

        return scheduler
    
    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()

class Config:
    pass
