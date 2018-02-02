'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import argparse
import numpy as np
import math
from collections import Counter
import sys
sys.path.append('../material')
from utils import load_pickle, dump_pickle


# Wikipedia-LSHTC: A=0.5,  B=0.4
# Amazon:          A=0.6,  B=2.6
# Other:		   A=0.55, B=1.5
def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('-A', '--A', type=float,
                       default=0.55,
                       help='A')
    parse.add_argument('-B', '--B', type=float,
                       default=1.5,
                       help='B')
    args = parse.parse_args()

    data_path = 'data/deeplearning_data/xml_data/'
    train_pid_label = load_pickle(data_path + 'train_label.pkl')

    index_label = load_pickle('data/baseline_data/xml_data/all_labels.pkl')
    baseline_inv_prop_file = 'data/baseline_data/xml_data/inv_prop.txt'

    train_label = train_pid_label.values()
    train_label = np.concatenate(train_label).tolist()
    label_frequency = dict(Counter(train_label))
    labels, fre = zip(*label_frequency.iteritems())
    fre = np.array(fre)

    N = len(train_pid_label)
    C = (math.log(N)-1) * (args.B + 1)**args.A
    inv_prop = 1 + C * (fre + args.B)**(-args.A)

    inv_prop_dict = dict(zip(labels, inv_prop.tolist()))
    dump_pickle(inv_prop_dict, data_path + 'inv_prop_dict.pkl')
    #
    # for baseline inv propensity
    with open(baseline_inv_prop_file, 'w') as df:
        for l_ in index_label[:-1]:
            df.write(str(inv_prop_dict[l_]))
            df.write('\n')
        df.write(str(inv_prop_dict[index_label[-1]]))



if __name__ == "__main__":
    main()
