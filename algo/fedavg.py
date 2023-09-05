from utils import *
from datasets import DS_INFO
from torch.multiprocessing import set_start_method, Pool
from operations import params_avg, train_clf, evaluate_clf, meta_update, mix_update

class FedAvg:
    def __init__(self, args, root_path, cpriv_sets, gtest_ds, gp_ds):
        self.args = args
        self.root_path = root_path
        self.cpriv_nums = [len(tset) for tset in cpriv_sets]
        logging.debug(f'cpriv_nums =  {self.cpriv_nums}')
        line = 'no proxy data splitted' if gp_ds is None else f'n_gp = {len(gp_ds)}'
        logging.debug(f'total_priv = {sum(self.cpriv_nums)}, {line}')

        self.tloaders = [DataLoader(tset, batch_size=args.bs_ltrain, shuffle=True) for tset in cpriv_sets]
        self.gtest_loader = DataLoader(gtest_ds, batch_size=args.bs_static, shuffle=False)

        self.n_classes = DS_INFO[args.dataset]['n_classes']
        self.img_shape = DS_INFO[args.dataset]['shape']

        feats_loc = -1 if 'feats_loc' not in args else args. feats_loc
        self.pmodel = create_model(args.arch, self.n_classes, self.img_shape, feats_loc).cuda()
        self.pre_gmodel = None

        self.iter_watcher = MetricsLogger(fpath=os.path.join(root_path, 'gmodel_metrics.csv'), title='i_iter,remark,n_epoch,gacc')
        self.best_tracker = BestTracker()
        if args.pool_size > 1:
            set_start_method('spawn')

    def local_training(self, cidx, gmodel):
        args = self.args
        cmodel = deepcopy(gmodel).cuda()
        optimizer = create_optim(cmodel.parameters(), args.lr_ltrain, args.optim_ltrain)
        scheduler = MySched(optimizer, args.ep_ltrain, args.sched_ltrain)
        mu_prox = args.mu_prox if 'mu_prox' in args else None
        wd_be = args.wd_be if 'wd_be' in args else None
        cmodel = train_clf(cmodel, self.tloaders[cidx], 
                    args.ep_ltrain, optimizer, scheduler,
                    gmodel=None if mu_prox is None else deepcopy(gmodel),
                    mu=mu_prox,
                    weight_decay=wd_be)
        return cmodel
    
    def local_extra(self, cidx, extra_base):
        return {}
    
    def local_steps(self, cidx, msg_brdcst:dict, i_iter=None):
        if i_iter: init_seed(i_iter)
        gmodel = msg_brdcst['gmodel']
        cmodel = self.local_training(cidx, gmodel)
        extra_base = {'cmodel': cmodel}
        msg_extra = self.local_extra(cidx, extra_base)
        msg = {'cidx': cidx, 'cmodel_params': cmodel.cpu().state_dict()}
        msg.update(msg_extra)
        return msg

    def local_update(self, cidxs, msg_brdcst:dict, i_iter):
        if self.args.pool_size > 0:
            p = Pool(self.args.pool_size)
            updated = []
            for cidx in cidxs:
                res = p.apply_async(func=self.local_steps,
                        args=(cidx, deepcopy(msg_brdcst), i_iter))
                updated.append(res)
            p.close()
            p.join()
            msg_list = [res.get() for res in updated]
        else:
            msg_list = [self.local_steps(cidx, deepcopy(msg_brdcst), i_iter)
                            for cidx in cidxs]

        return msg_list

    def parameter_averaging(self, msg_list, i_iter=None, weights=None):
        w = self.cpriv_nums if weights is None else weights
        w_params = [(w[msg['cidx']], msg['cmodel_params']) 
                        for msg in msg_list]
        avg_params = params_avg(w_params)
        gmodel = deepcopy(self.pmodel)
        gmodel.load_state_dict(avg_params)
        gmodel = gmodel.cuda()

        gacc = None
        if i_iter is not None:
            gacc, _ = evaluate_clf(gmodel, self.gtest_loader)
            mode = self.args.ep_ltrain if weights is None else 'weighted'
            self.iter_watcher.log(f'{i_iter},param_avg,{mode},{gacc:.4f}')
        
        return gmodel, gacc
    
    def build_cmodels_dict(self, msg_list):
        cmodels_dict = {}
        for msg in msg_list:
            cidx = msg['cidx']
            cmodel = deepcopy(self.pmodel)
            cmodel.load_state_dict(msg['cmodel_params'])
            cmodel = cmodel.cuda()
            cmodel.eval()
            cmodels_dict[cidx] = cmodel

        return cmodels_dict
        
    def aggregate(self, msg_list, i_iter):
        gmodel, gacc = self.parameter_averaging(msg_list, i_iter)
        return gmodel, gacc
    
    def update_gmodel(self, gmodel, gacc, i_iter=None):
        args = self.args
        if args.lr_meta is not None:
            gmodel = meta_update(self.pre_gmodel, gmodel, args.lr_meta)
            update_mode = 'meta'
            hyper = args.lr_meta
        elif args.beta is not None:
            gmodel = mix_update(self.pre_gmodel, gmodel, args.beta)
            update_mode = 'beta'
            hyper = args.beta
        
        cond1 = args.lr_meta is not None or args.beta is not None
        cond2 = i_iter is not None
        if cond1 and cond2:
            gacc, _ = evaluate_clf(gmodel, self.gtest_loader)
            self.iter_watcher.log(f'{i_iter},{update_mode},{hyper},{gacc:.4f}')
        
        return gmodel, gacc
    
    def server_update(self, msg_list, i_iter):
        gmodel, gacc = self.aggregate(msg_list, i_iter)
        gmodel, gacc = self.update_gmodel(gmodel, gacc, i_iter)
        msg_brdcst = {'gmodel': gmodel}
        return msg_brdcst, gacc
    
    def process_watching(self, gacc, i_iter):
        self.best_tracker.watch(gacc, i_iter)
        if i_iter % self.args.freq == 0:
            self.iter_watcher.print()            
            best_gacc, best_iter = self.best_tracker.get_best()
            logging.debug(f'gacc={gacc:.4f}, best_gacc={best_gacc:.4f} at {best_iter:03d}/{i_iter:03d}')
        
        if i_iter == self.args.n_iter:
            best_gacc, best_iter = self.best_tracker.get_best()
            logging.debug(f'After {self.args.n_iter:03d} iters, best_gacc={best_gacc:.4f} at {best_iter:03d}')
    
    def train(self):
        args = self.args
        msg_brdcst = {'gmodel': deepcopy(self.pmodel)}
        n_active = round(args.n_clients * args.r_active)

        count = 0
        bad_gacc = 1 / self.n_classes + 1e-6
        for i_iter in range(1, args.n_iter+1):
            self.pre_gmodel = deepcopy(msg_brdcst['gmodel'])
            cidxs = client_sampling(i_iter, args.n_clients, n_active)
            msg_list = self.local_update(cidxs, msg_brdcst, i_iter)
            msg_brdcst, gacc = self.server_update(msg_list, i_iter)

            self.process_watching(gacc, i_iter)

            if gacc < bad_gacc: count += 1
            else: count = 0
            if count >= 10: break