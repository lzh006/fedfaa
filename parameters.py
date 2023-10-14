import argparse

def int_list(e):
    return None if e is None else eval(e)

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='lenet', help='lenet; resnetX')
    parser.add_argument('--dataset', type=str, default='fashion')
    parser.add_argument('--n_grdd', type=int, default=None)
    parser.add_argument('--n_gp', type=int, default=10000, help='10k in paper FedBE')

    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--r_active', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--bs_ltrain', type=int, default=64)
    parser.add_argument('--ep_ltrain', type=int, default=20)
    parser.add_argument('--lr_ltrain', type=float, default=0.01)
    parser.add_argument('--optim_ltrain', type=str, default='adam')
    parser.add_argument('--sched_ltrain', type=str, default=None)
    parser.add_argument('--aug_ltrain', action='store_true')

    parser.add_argument('--lr_meta', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)

    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--bs_static", type=int, default=8192)
    parser.add_argument('--pool_size', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--rpid', type=str, default=None)
    parser.add_argument('--timestamp', type=str, default=None)

    subparsers = parser.add_subparsers(dest='algo')

    parser_fedprox = subparsers.add_parser('fedavg')

    parser_fedprox = subparsers.add_parser('fedprox')
    parser_fedprox.add_argument('--mu_prox', type=float, default=0.01, help='usually: [0.001, 0.01, 0.1, 1]. 0.01, 0.001, and 0.001 for CIFAR-10, CIFAR-100, and Tiny-Imagenet reported by MOON')

    parser_feddf = subparsers.add_parser('feddf')
    parser_feddf.add_argument('--dn_publ', type=str, default=None)
    parser_feddf.add_argument('--bs_kd', type=int, default=256)
    parser_feddf.add_argument('--ep_kd', type=int, default=20, help='20 by paper FedCG')
    parser_feddf.add_argument('--lr_kd', type=float, default=0.001, help='1e-3 wich cos in paper')
    parser_feddf.add_argument('--T_kd', type=float, default=20.0)
    parser_feddf.add_argument('--optim_kd', type=str, default='adam')
    parser_feddf.add_argument('--sched_kd', type=str, default='cos')

    parser_fedbe = subparsers.add_parser('fedbe')
    parser_fedbe.add_argument('--dn_publ', type=str, default=None)
    parser_fedbe.add_argument('--bs_kd', type=int, default=256)
    parser_fedbe.add_argument('--ep_kd', type=int, default=20, help='20 in paper')
    parser_fedbe.add_argument('--lr_kd', type=float, default=0.001, help='1e-3 --> 1e-4 in paper')
    parser_fedbe.add_argument('--T_kd', type=float, default=20.0)
    parser_fedbe.add_argument('--optim_kd', type=str, default='adam')
    parser_fedbe.add_argument('--sched_kd', type=str, default='cos')

    parser_fedbe.add_argument('--n_steacher', type=int, default=10, help='M=10 in paper')
    parser_fedbe.add_argument('--wd_be', type=float, default=None, help='0.0005 in paper')

    parser_fedftg = subparsers.add_parser('fedftg')
    parser_fedftg.add_argument('--bs_kd', type=int, default=256)
    parser_fedftg.add_argument('--ep_kd', type=int, default=20)
    parser_fedftg.add_argument('--lr_kd', type=float, default=0.001)
    parser_fedftg.add_argument('--T_kd', type=float, default=20.0)
    parser_fedftg.add_argument('--optim_kd', type=str, default='adam')
    parser_fedftg.add_argument('--sched_kd', type=str, default='cos')

    parser_fedftg.add_argument('--arch_gen', type=str, default='fast', help='the gen DFAD used in paper')
    parser_fedftg.add_argument('--ngf', type=int, default=8)
    parser_fedftg.add_argument('--nz', type=int, default=256, help='256 in paper')

    parser_fedftg.add_argument('--eta_g', type=float, default=0.05, help='not provided in paper')
    parser_fedftg.add_argument('--st_i', type=int, default=10, help='I=10 in paper as fine-tuning iterations')
    parser_fedftg.add_argument('--st_g', type=int, default=1, help='I_g=1 in paper as gen-training steps')
    parser_fedftg.add_argument('--st_d', type=int, default=5, help='I_d=5 in paper as kd steps for fine-tuning')
    parser_fedftg.add_argument('--clf', type=float, default=1, help='lambda_clf=1.0 in paper')
    parser_fedftg.add_argument('--divr', type=float, default=1, help='lambda_dis=1.0 in paper')
    parser_fedftg.add_argument('--modify', action='store_true', help='wrong updating eq in algo, but seems correct in text')
    parser_fedftg.add_argument('--es_probe', action='store_true', help='just used for hyperparameter searching, tune off in application')

    parser_fedfaa = subparsers.add_parser('fedfaa')
    parser_fedfaa.add_argument('--dn_publ', type=str, default=None)
    parser_fedfaa.add_argument('--bs_kd', type=int, default=256)
    parser_fedfaa.add_argument('--ep_kd', type=int, default=20, help='20 by paper FedCG')
    parser_fedfaa.add_argument('--lr_kd', type=float, default=0.001, help='1e-3 wich cos in paper')
    parser_fedfaa.add_argument('--T_kd', type=float, default=20.0)
    parser_fedfaa.add_argument('--optim_kd', type=str, default='adam')
    parser_fedfaa.add_argument('--sched_kd', type=str, default='cos')

    parser_fedfaa.add_argument('--feats_loc', type=int, default=-1)
    parser_fedfaa.add_argument('--r_ceil', type=float, default=0.75)
    parser_fedfaa.add_argument('--theta_tune', type=float, default=0.5)
    parser_fedfaa.add_argument('--mode_r', type=str, default='constant', help='constant, decay, inc')
    parser_fedfaa.add_argument('--mode_theta', type=str, default='decay', help='constant, decay, inc')
    parser_fedfaa.add_argument('--deci_mu', type=int, default=6)

    return parser.parse_args()
