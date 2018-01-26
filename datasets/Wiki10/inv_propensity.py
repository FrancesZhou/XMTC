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
                       default=0.5,
                       help='A')
    parse.add_argument('-B', '--B', type=float,
                       default=0.4,
                       help='B')
    args = parse.parse_args()

    data_path = 'data/deeplearning_data/adjacent_labels/all_para/'
    train_pid_label = load_pickle(data_path + 'train_label.pkl')

    train_label = train_pid_label.values()
    train_label = np.concatenate(train_label).tolist()
    label_frequency = dict(Counter(train_label))
    labels = label_frequency.keys()
    fre = np.array(label_frequency.values())

    N = len(train_pid_label)
    C = (math.log(N)-1) * (args.B + 1)**args.A
    inv_prop = 1 + C * (fre + args.B)**(-args.A)

    inv_prop_dict = dict(zip(labels, inv_prop.tolist()))
    dump_pickle(inv_prop_dict, data_path + 'inv_prop_dict.pkl')



if __name__ == "__main__":
    main()
