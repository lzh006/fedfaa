from parameters import args_parser
from utils import *

def main(args, root_path):
    cpriv_sets, gtest_ds, gp_ds = load_fed_data(args, data_dir='brenchmarks', root_path=root_path)
    ALGRS = {'fedavg': FedAvg, 'fedhkf': FedHKF}
    fed = ALGRS[args.algo](args, root_path, cpriv_sets, gtest_ds, gp_ds)
    fed.train()

if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.algo is None: args.algo = 'fedavg'
    root_path, timestamp = create_log_dir(args.algo, timestamp=args.timestamp)
    logging.debug(f'timestamp = {timestamp}, PID = {os.getpid()}, seed={args.seed}, gpu={args.gpu}')
    logging.debug(f'args={args}')

    init_seed(args.seed)

    from algo import *
    main(args, root_path)
