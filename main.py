from adaboost import adaboost, getRecList
from eval import get_recall
import json
import pickle
import magic
import argparse
import sys
from scipy.sparse import load_npz


class readFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, values)
        else:
            FileType = magic.from_file(values)
            if "Zip" in FileType:
                setattr(namespace, self.dest, load_npz(values))
            elif "JSON" in FileType:
                setattr(namespace, self.dest, json.load(open(values)))
            elif "data" in FileType:
                setattr(namespace, self.dest, pickle.load(open(values, "rb")))
            else:
                raise TypeError("file type not allowed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='''
    main.py train  -d <training_data> [--save_time] [-i iteration]
            test   -d <test_data> -m <model> -r <recall_at> [recall_at ...] [-od opt_data] [-tu test_users] [-ai available_items]
            rec    -m <model> -n <num_rec> [-od opt_data] [-tu test_users] [-ai available_items] '''
    )
    parser.add_argument('command', choices=["train", "test", "rec"])
    parser.add_argument('-d', '--data', required='test' in sys.argv or 'train' in sys.argv,
                        action=readFile, help='csr_matrix in npz file')
    parser.add_argument('-m', '--model', required='rec' in sys.argv or 'test' in sys.argv,
                        action=readFile, help='List of models, load with pickle')
    parser.add_argument('-n', '--n_rec', type=int,
                        required='rec' in sys.argv, help='# of recommanded items per user')
    parser.add_argument('-r', '--recall_at', required='test' in sys.argv,
                        nargs="+", type=int, help='Evaluate recall at top r (starting from 0)')

    parser.add_argument('--save_time', action='store_true',
                        help='Specify whether to save ram or time when training.')
    parser.add_argument('--in_out', action='store_true',
                        help='Whether to test in/out of matrix seperately')
    parser.add_argument('-ni', '--n_iter', type=int,
                        default=2, help='# of iterations of adaboost')
    parser.add_argument('-od', '--opt_data', action=readFile,
                        help='Optional data:csr_matrix in npz file ')
    parser.add_argument('-tu', '--test_users', action=readFile,
                        help='Indexes in the trained model. Type = list, in json file')
    parser.add_argument('-ai', '--available_items', default=slice(None), action=readFile,
                        help='Indexes in the trained model. Type = list, in json file')
    # parser.add_argument('-bs','--batch_size', type=int, default=5000, help='Size of user batch')
    # parser.add_argument('-th','--thread', type=int, default=6, help='# of threads for multi processing')
    args = parser.parse_args()

    if args.command == "train":
        ensemble = adaboost(args.opt_data, args.data, args.n_iter, saveTime=args.save_time, modelList=args.model)  # retrain:ensemble=ensemble
        with open("data/ensemble2", "wb")as f:
            pickle.dump(ensemble, f)
    elif args.command == "test":
        recall = get_recall(args.model, args.data, args.opt_data, args.recall_at, args.test_users, args.available_items, args.in_out)
        print(recall)
    elif args.command == "rec":
        topidx = getRecList(args.model, args.n_rec, args.opt_data, args.test_users, args.available_items)
        print(topidx.shape)
