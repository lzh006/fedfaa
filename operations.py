from utils import *
from datasets import BasicDataset, insert_batch
from torch.nn.utils import clip_grad_norm_
from torch.nn import Module, CosineSimilarity, CrossEntropyLoss, functional as F


@torch.no_grad()
def evaluate_clf(model, vloader):
    model = model.cuda()
    model.eval()
    criterion = CrossEntropyLoss().cuda()
    n_total, macc, mloss = 0, 0, 0
    for inputs, targets in vloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs, -1)
        mask_corrs = predicted.eq(targets)

        bs = targets.size(0)
        n_total += bs
        mloss += loss.item() * bs
        macc += mask_corrs.sum().item()

    return macc/n_total, mloss/n_total

def train_clf(model, tloader, epoch, optimizer, scheduler, gmodel=None, mu=None, weight_decay=None):
    model.train()
    ce = CrossEntropyLoss().cuda()
    gw = None if gmodel is None else list(gmodel.cuda().parameters())
    for _ in range(epoch):
        for inputs, targets in tloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ce(outputs, targets)

            if gw is not None:
                fed_prox_reg = 0.0
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - gw[param_index])) ** 2)
                loss += fed_prox_reg

            loss.backward()
            clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.0)
            if weight_decay is not None:
                pass
                
            optimizer.step()
            scheduler.step()

    return model

def params_avg(w_params):
    training_num = 0
    for idx in range(len(w_params)):
        (sample_num, averaged_params) = w_params[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_params[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_params)):
            local_sample_number, local_model_params = w_params[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params

def meta_update(pre_gmodel, gmodel, lr_meta):
    meta_optimizer = create_optim(pre_gmodel.parameters(), lr_meta, 'adam')
    pre_gmodel.point_grad_to(gmodel)
    meta_optimizer.step()
    return pre_gmodel
    
def mix_update(pre_gmodel, gmodel, beta):
    mix_gmodel = deepcopy(gmodel)
    w_params = [(1-beta, pre_gmodel.cpu().state_dict()),(beta, gmodel.cpu().state_dict())]
    avg_params = params_avg(w_params)
    mix_gmodel.load_state_dict(avg_params)
    mix_gmodel = mix_gmodel.cuda()
    return mix_gmodel


@torch.no_grad()
def selective_guidance_0(tmodels_dict, midx_list, mask_s, ploader, i_x=None):
    n_classes = tmodels_dict[midx_list[0]].n_classes
    device = torch.device('cuda')
    
    kd_ds = BasicDataset()
    i_start = 0
    for samples in ploader:
        x = samples.cuda() if i_x is None else samples[i_x].cuda()
        bs = x.size(0)
        logits = torch.zeros(size=(bs, n_classes), device=device)
        
        indexs = np.arange(i_start, i_start+bs)
        i_start += bs
        mask_candi = mask_s[indexs]
        indices = np.where(np.all(mask_candi == False, axis=0))
        mask_tiny = torch.from_numpy(np.delete(mask_candi, indices, axis=1)).cuda()
        candidates = np.delete(midx_list.copy(), indices)
        for i in range(len(candidates)):
            mask_t = mask_tiny[:,i]
            logits[mask_t] += tmodels_dict[candidates[i]](x[mask_t])
        
        nums = mask_tiny.sum(dim=1, keepdim=True)
        logits = logits / nums
        kd_ds.insert([x, logits])
        
    return kd_ds


@torch.no_grad()
def create_guidance_0(tmodels_dict, ploader, i_target=None, qa=False):
    counts_hit = {}
    for cidx in tmodels_dict:
        tmodels_dict[cidx] = tmodels_dict[cidx].cuda()
        tmodels_dict[cidx].eval()
        counts_hit[cidx] = 0
    
    cidxs = list(tmodels_dict.keys())
    n_classes = tmodels_dict[cidxs[0]].n_classes
    n_refs = len(tmodels_dict)
    device = torch.device('cuda')
    
    kd_ds = BasicDataset()
    for samples in ploader:
        inputs = samples.cuda() if i_target is None else samples[0].cuda()
        targets = None if i_target is None else samples[i_target].cuda()
        bs = inputs.size(0)
        logits_t = torch.zeros(size=(bs,n_classes), device=device)
        logits_part = torch.zeros(size=(bs,n_classes), device=device)
        nums = torch.zeros(bs, device=device)

        for cidx in tmodels_dict:
            cmodel = tmodels_dict[cidx]
            logits_c = cmodel(inputs)
            logits_t += logits_c

            if qa:
                assert targets is not None
                preds = torch.max(logits_c, -1)[1]
                mcorr = preds.eq(targets) | (targets >= n_classes)
                logits_part[mcorr] += logits_c[mcorr]
                nums += mcorr
                counts_hit[cidx] += mcorr.sum().item()
            else:
                nums += 1
                counts_hit[cidx] += bs
        
        if qa:
            mcap = nums > 0
            logits_t[mcap] = logits_part[mcap]
            nums[~mcap] = n_refs
        
        logits_t = logits_t / nums.view(-1, 1)
        kd_ds.insert([inputs, logits_t])
    
    return kd_ds, counts_hit


@torch.no_grad()
def create_guidance(tmodels_dict, ploader, i_target=None, qa=False):
    counts_hit = {}
    for cidx in tmodels_dict:
        tmodels_dict[cidx] = tmodels_dict[cidx].cuda()
        tmodels_dict[cidx].eval()
        counts_hit[cidx] = 0
    
    cidxs = list(tmodels_dict.keys())
    n_classes = tmodels_dict[cidxs[0]].n_classes
    n_refs = len(tmodels_dict)
    device = torch.device('cuda')

    total_ce_loss = 0
    sample_count = 0
    corret_teacher_count = 0
    total_teacher_count = 0
    ce = CrossEntropyLoss().cuda()
    
    kd_ds = BasicDataset()
    for samples in ploader:
        inputs = samples.cuda() if i_target is None else samples[0].cuda()
        targets = None if i_target is None else samples[i_target].cuda()
        bs = inputs.size(0)
        logits_t = torch.zeros(size=(bs,n_classes), device=device)
        logits_part = torch.zeros(size=(bs,n_classes), device=device)
        nums = torch.zeros(bs, device=device)

        for cidx in tmodels_dict:
            cmodel = tmodels_dict[cidx]
            logits_c = cmodel(inputs)
            logits_t += logits_c

            tea_preds = torch.max(logits_c, -1)[1]
            tea_mcorr = tea_preds.eq(targets)
            corret_teacher_count += tea_mcorr.sum().item()
            total_teacher_count += logits_c.size(0)

            if qa:
                assert targets is not None
                preds = torch.max(logits_c, -1)[1]
                mcorr = preds.eq(targets) | (targets >= n_classes)
                logits_part[mcorr] += logits_c[mcorr]
                nums += mcorr
                counts_hit[cidx] += mcorr.sum().item()
            else:
                nums += 1
                counts_hit[cidx] += bs
        
        if qa:
            mcap = nums > 0
            logits_t[mcap] = logits_part[mcap]
            nums[~mcap] = n_refs
        
        logits_t = logits_t / nums.view(-1, 1)
        kd_ds.insert([inputs, logits_t])

        total_ce_loss += ce(logits_t, targets).item() * bs
        sample_count += bs
    
    return kd_ds, counts_hit, total_ce_loss/sample_count, corret_teacher_count/total_teacher_count

class KLDiv(Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, s_logits, t_logits):
        T = self.T
        q = F.log_softmax(s_logits / T, dim=1)
        p = F.softmax(t_logits / T, dim=1 )
        return F.kl_div(q, p, reduction=self.reduction) * (T*T)

def kd_epochs(model, kd_loader, epoch, optimizer, scheduler, T=20.0, use_SWA=False):
    model = model.cuda()
    model.train()
    kl = KLDiv(T).cuda()
    for _ in range(epoch):
        for inputs, logits_t in kd_loader:
            inputs, logits_t = inputs.cuda(), logits_t.cuda()
            optimizer.zero_grad()
            logits_s = model(inputs)
            loss = kl(logits_s, logits_t)

            loss.backward()
            clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.0)
            optimizer.step()
            scheduler.step()
    
    if use_SWA:
        pass
    
    return model


class DiversityLoss(Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric='l1'):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

def cat_array(obj, batch_data):
    if obj is None: return batch_data
    if batch_data is None or len(batch_data) == 0: return obj
    return np.concatenate([obj, batch_data], axis=0)

@torch.no_grad()
def extract_feats(model, dloader, i_x=None, i_y=None, label_list=None, use_rep=True):
    model = model.cuda()
    model.eval()
    feats_dict, indexs_dict = {}, {}
    i_start = 0
    for sample in dloader:
        x = sample.cuda() if i_x is None else sample[i_x].cuda()
        logits, feats = model(x, feats_mode='also')
        if not use_rep: feats = logits
        preds = torch.max(logits, -1)[1]
        bs = x.size(0)
        indexs = np.arange(i_start, i_start+bs)
        i_start += bs

        if i_y is not None:
            y = sample[i_y].cuda()
            mask_corr = preds.eq(y)
            if mask_corr.any():
                preds = preds[mask_corr]
                feats = feats[mask_corr]
                indexs = indexs[mask_corr.cpu().numpy()]
            else:
                preds = None
        
        if preds is not None:
            for c in preds.unique():
                label = int(c)
                if label_list is not None and label not in label_list: continue

                mask = preds == c
                feats_dict.setdefault(label, None)
                cfeats = feats[mask]
                feats_dict[label] = insert_batch(feats_dict[label], cfeats)
                
                cindexs = indexs[mask.cpu().numpy()]
                indexs_dict.setdefault(label, None)
                indexs_dict[label] = cat_array(indexs_dict[label], cindexs)
                
    return feats_dict, indexs_dict

@torch.no_grad()
def selective_guidance(tmodels_dict, midx_list, mask_s, ploader, i_x=None):
    n_classes = tmodels_dict[midx_list[0]].n_classes
    device = torch.device('cuda')
    
    kd_ds = BasicDataset()
    i_start = 0
    total_ce_loss = 0
    sample_count = 0
    corret_teacher_count = 0
    total_teacher_count = 0
    ce = CrossEntropyLoss()
    for x, y in ploader:
        x, y = x.cuda(), y.cuda()
        bs = x.size(0)
        logits = torch.zeros(size=(bs, n_classes), device=device)
        
        indexs = np.arange(i_start, i_start+bs)
        i_start += bs
        mask_candi = mask_s[indexs]
        indices = np.where(np.all(mask_candi == False, axis=0))
        mask_tiny = torch.from_numpy(np.delete(mask_candi, indices, axis=1)).cuda()
        candidates = np.delete(midx_list.copy(), indices)
        for i in range(len(candidates)):
            mask_t = mask_tiny[:,i]
            tea_logits = tmodels_dict[candidates[i]](x[mask_t])
            logits[mask_t] += tea_logits

            tea_preds = torch.max(tea_logits, -1)[1]
            tea_mcorr = tea_preds.eq(y[mask_t])
            corret_teacher_count += tea_mcorr.sum().item()
            total_teacher_count += tea_logits.size(0)
        
        nums = mask_tiny.sum(dim=1, keepdim=True)
        logits = logits / nums
        kd_ds.insert([x, logits])

        total_ce_loss += ce(logits, y).item() * bs
        sample_count += bs
        
    return kd_ds, total_ce_loss/sample_count, corret_teacher_count/total_teacher_count



@torch.no_grad()
def extract_logits(model, dloader, i_x=None, i_y=None, label_list=None):
    model = model.cuda()
    model.eval()
    logits_dict, indexs_dict = {}, {}
    i_start = 0
    for sample in dloader:
        x = sample.cuda() if i_x is None else sample[i_x].cuda()
        logits, feats = model(x, feats_mode='also')
        preds = torch.max(logits, -1)[1]
        bs = x.size(0)
        indexs = np.arange(i_start, i_start+bs)
        i_start += bs

        if i_y is not None:
            y = sample[i_y].cuda()
            mask_corr = preds.eq(y)
            if mask_corr.any():
                preds = preds[mask_corr]
                logits = logits[mask_corr]
                indexs = indexs[mask_corr.cpu().numpy()]
            else:
                preds = None
        
        if preds is not None:
            for c in preds.unique():
                label = int(c)
                if label_list is not None and label not in label_list: continue

                mask = preds == c
                logits_dict.setdefault(label, None)
                clogits = logits[mask]
                logits_dict[label] = insert_batch(logits_dict[label], clogits)
                
                cindexs = indexs[mask.cpu().numpy()]
                indexs_dict.setdefault(label, None)
                indexs_dict[label] = cat_array(indexs_dict[label], cindexs)
                
    return logits_dict, indexs_dict
