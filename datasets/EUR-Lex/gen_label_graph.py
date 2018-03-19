'''
Created on Mar, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
from ..material.utils import load_pickle, dump_pickle, load_txt, read_label_pairs
# import sys
# sys.path.append('../material')
#from utils import load_pickle, dump_pickle, load_txt, read_label_pairs

data_source_path = 'data/deeplearning_data/docs/xml_data/'
data_des_path = 'data/deeplearning_data/xml_data/'

def gen_graph():
    graph = read_label_pairs(data_des_path + 'labels.edgelist')
    return graph

#def gen_weighted_graph():

def main():
    gen_graph()


if __name__ == "__main__":
    main()