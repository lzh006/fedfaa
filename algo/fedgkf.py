from utils import *
from .fedavg import FedAvg
from operations import extract_feats, kd_epochs, selective_guidance, params_avg, evaluate_clf, create_guidance, selective_guidance_0, create_guidance_0
import nmslib

class FedGKF(FedAvg):
    def __init__(self, args, root_path, cpriv_sets, gtest_ds, gp_ds):
        super().__init__(args, root_path, cpriv_sets, gtest_ds, gp_ds)
        self.publ_loader = DataLoader(gp_ds, batch_size=args.bs_static, shuffle=False)
        self.n_publ = len(gp_ds)
        dn_publ = args.dn_publ if 'dn_publ' in args and args.dn_publ is not None else args.dataset
        logging.debug(f'proxy data: public {dn_publ} = {self.n_publ}')

        self.cpriv_loaders = [DataLoader(ds, batch_size=args.bs_static, shuffle=False) for ds in cpriv_sets]
        self.sconfig = self.set_searching_params()
        self.cache_models = {}
        self.cache_params = {}
    
    def kd(self, kd_ds, gmodel, i_iter):
        args = self.args
        optimizer = create_optim(gmodel.parameters(), args.lr_kd, args.optim_kd)
        scheduler = MySched(optimizer, args.ep_kd, args.sched_kd)
        kd_loader = DataLoader(kd_ds, batch_size=args.bs_kd, shuffle=True)
        gmodel = kd_epochs(gmodel, kd_loader, args.ep_kd, optimizer, scheduler, T=args.T_kd)

        gacc = None
        if i_iter is not None:
            gacc, _ = evaluate_clf(gmodel, self.gtest_loader)
            self.iter_watcher.log(f'{i_iter},kd,{args.ep_kd},{gacc:.4f}')
        
        return gmodel, gacc

    def aggregate(self, msg_list, i_iter):
        avg_model, _ = self.parameter_averaging(msg_list, i_iter)

        tmodels_dict = self.build_cmodels_dict(msg_list)
        kd_ds, _ = create_guidance_0(tmodels_dict, self.publ_loader, 1)
        # kd_ds, _, kd_ce, corr_tea_rate = create_guidance(tmodels_dict, self.publ_loader, 1)
        # self.iter_watcher.log(f'{i_iter},ce-ctr,{kd_ce:.6f},{corr_tea_rate:.4f}')
        gmodel, gacc = self.kd(kd_ds, avg_model, i_iter)
        
        return gmodel, gacc
    
    def set_searching_params(self):
        config = Config()
        config.M = 15
        config.efC = 100
        config.num_threads = 16
        config.K = 1
        config.efS = 100
        config.query_time_params = {'efSearch': config.efS}
        config.index_time_params = {'M': config.M, 'indexThreadQty': config.num_threads, 'efConstruction': config.efC, 'post': 0}
        config.space_name = 'cosinesimil'
        config.method = 'hnsw'
        logging.debug(f'searching parameters: {config.__dict__}')
        return config

    def minimal_sim_dists(self, feats_dict_p, indexs_dict_p, feats_dict_c):
        dists = np.ones(self.n_publ)
        config = self.sconfig
        for c in feats_dict_p:
            if c in feats_dict_c:
                index = nmslib.init(method=config.method,
                                    space=config.space_name,
                                    data_type=nmslib.DataType.DENSE_VECTOR)
                index.addDataPointBatch(feats_dict_c[c])
                index.createIndex(config.index_time_params)
                index.setQueryTimeParams(config.query_time_params)
                nbrs = index.knnQueryBatch(feats_dict_p[c], k=config.K, num_threads=config.num_threads)
                dists[indexs_dict_p[c]] = np.array([sim_dist[0] for _, sim_dist in nbrs])

        return dists

    def local_extra(self, cidx, extra_base):
        sim_dists = {}
        for midx in extra_base['finfo']:
            model = extra_base[midx]
            feats_dict_p, indexs_dict_p = extra_base['finfo'][midx]
            feats_dict_c, _ = extract_feats(model, self.cpriv_loaders[cidx], i_x=0, i_y=1, 
                                            label_list=list(feats_dict_p.keys()))
            sim_dists[midx] = self.minimal_sim_dists(feats_dict_p, indexs_dict_p, feats_dict_c)
        
        return {'sim_dists': sim_dists}
    
    def local_steps(self, cidx, msg_brdcst:dict, i_iter=None):
        if i_iter: init_seed(i_iter)
        gmodel = msg_brdcst['gmodel']
        cmodel = self.local_training(cidx, gmodel)

        extra_base = {e:msg_brdcst[e] for e in msg_brdcst if isinstance(e, int)}
        extra_base[-1] = gmodel
        extra_base[cidx] = cmodel
        extra_base['finfo'] = msg_brdcst['finfo'] if 'finfo' in msg_brdcst else {}
        extra_base['finfo'][cidx] = extract_feats(cmodel, self.publ_loader, i_x=0)
        msg_extra = self.local_extra(cidx, extra_base)

        msg = {'cidx': cidx, 'cmodel_params': cmodel.cpu().state_dict()}
        msg.update(msg_extra)
        return msg

    def summarize_dist_table(self, msg_list):
        dist_dict = {}
        for msg in msg_list:
            sim_dists = msg['sim_dists']
            for midx in sim_dists:
                dist_dict.setdefault(midx,[])
                dist_dict[midx].append(sim_dists[midx])
        
        cache = []
        midx_list = list(dist_dict.keys())
        for midx in midx_list:
            v_matrix = np.stack(dist_dict[midx])
            cache.append(np.min(v_matrix, axis=0))
        
        return np.stack(cache).T, midx_list

    def cal_p(self, dist_table, mu, r_ceil):
        mask_q = dist_table <= mu
        row_r = mask_q.sum(axis=1) / dist_table.shape[1]
        return sum(row_r > r_ceil) / dist_table.shape[0]

    def tune_mu(self, dist_table, r_ceil, theta, deci=2):
        left = np.min(dist_table)
        right = np.max(dist_table)
        while left <= right:
            mu = round((left + right) / 2, deci)
            p = self.cal_p(dist_table, mu, r_ceil)
            # logging.debug(f'[tune] mu={mu:.5f}, left={left:.5f}, right={right:.5f}, p={p:.4f}, theta={theta:.4f}')
            if p < theta: left = mu + 1 / 10**deci
            else: right = mu - 1 / 10**deci
        return right

    def mark_selection(self, dist_table, mu):
        mask_s = dist_table <= mu
        for i in range(dist_table.shape[0]):
            if mask_s[i].sum() == 0:
                mask_s[i] = np.ones(dist_table.shape[1])
        return mask_s

    def k_smallest_in_rows(self, A, k):
        result = np.zeros_like(A, dtype=bool)
        for i, row in enumerate(A):
            result[i] = row <= np.partition(row, k-1)[k-1]
        return result

    def adoption_avg(self, msg_list, weights, i_iter=None):
        w_params = [(weights[msg['cidx']], msg['cmodel_params']) for msg in msg_list]
        for midx in self.cache_params:
            w_params.append((weights[midx], self.cache_params[midx]))
        
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
    
    def gradu_param(self, v_target, mode, i_iter, n_iter, v_max=1.0):
        if mode == 'decay':
            v = v_max - (v_max-v_target)/(n_iter-1)*(i_iter-1)
        elif mode == 'inc':
            v = v_target + (v_max-v_target)/(n_iter-1)*(i_iter-1)
        else:
            v = v_target
        
        return v

    def aggregate(self, msg_list, i_iter):
        args = self.args

        dist_table, midx_list = self.summarize_dist_table(msg_list)

        if args.mode_theta == 'decay':
            theta = 1 - (1-args.theta_tune)/(args.n_iter-1)*(i_iter-1)
        elif args.mode_theta == 'inc':
            theta = args.theta_tune + (1-args.theta_tune)/(args.n_iter-1)*(i_iter-1)
        else:
            theta = args.theta_tune

        r_ceil = self.gradu_param(v_target=args.r_ceil, 
                                        mode=args.mode_r, 
                                        i_iter=i_iter, n_iter=args.n_iter)
        theta = self.gradu_param(v_target=args.theta_tune, 
                                        mode=args.mode_theta, 
                                        i_iter=i_iter, n_iter=args.n_iter)
        mu = self.tune_mu(dist_table, r_ceil, theta, deci=args.deci_mu)
        mask_s = self.mark_selection(dist_table, mu)


        nums_adoption = mask_s.sum(axis=0)
        weights = {midx_list[i]:nums_adoption[i] for i in range(len(midx_list))}

        avg_model, _ = self.adoption_avg(msg_list, weights, i_iter)

        tmodels_dict = self.build_cmodels_dict(msg_list)
        tmodels_dict.update(self.cache_models)

        kd_ds = selective_guidance_0(tmodels_dict, midx_list, mask_s, self.publ_loader, i_x=0)

        gmodel, gacc = self.kd(kd_ds, avg_model, i_iter)

        return gmodel, gacc

    def server_action(self, gmodel):
        args = self.args
        msg_extra = {'finfo':{}}

        msg_extra['finfo'][-1] = extract_feats(gmodel, self.publ_loader, i_x=0)
        self.cache_models.clear()
        self.cache_models[-1] = deepcopy(gmodel)
        self.cache_params.clear()
        self.cache_params[-1] = deepcopy(gmodel).cpu().state_dict()

        return msg_extra

    def server_update(self, msg_list, i_iter):
        gmodel, gacc = self.aggregate(msg_list, i_iter)
        gmodel, gacc = self.update_gmodel(gmodel, gacc, i_iter)
        msg_brdcst = {'gmodel': gmodel}

        msg_extra = self.server_action(gmodel)
        msg_brdcst.update(msg_extra)

        return msg_brdcst, gacc