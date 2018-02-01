'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import
import numpy as np


# if len(np.intersect1d(pre_labels, true_labels)):
#     count += 1
# return count*1.0/num
# return count, count * 1.0 / num

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k, true_num=5):
    #dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    dcg_max = dcg_at_k(np.ones(k), min(k, true_num))
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def precision(pre, tar, indices):
    p_1 = []
    p_3 = []
    p_5 = []
    ndcg_1 = []
    ndcg_3 = []
    ndcg_5 = []
    for i in range(len(pre)):
        # pre_labels = np.argsort(pre[i])
        label_score = []
        s_indices = np.array(indices[i])
        s_pre = pre[i]
        for k in range(len(s_indices)):
            label_score.append((s_indices[k], s_pre[k]))
        label_score = sorted(label_score, key=lambda x :x[1])
        label_score.reverse()
        pre_labels = [x[0] for x in label_score]
        true_pos = np.squeeze(np.nonzero(tar[i]))
        true_labels = np.array(s_indices[true_pos])
        r = []
        for p_ in pre_labels:
            if p_ in true_labels:
                r.append(1)
            else:
                r.append(0)
        p_1.append(np.mean(r[:1]))
        p_3.append(np.mean(r[:3]))
        p_5.append(np.mean(r[:5]))
        ndcg_1.append(ndcg_at_k(r, 1))
        ndcg_3.append(ndcg_at_k(r, 3))
        ndcg_5.append(ndcg_at_k(r, 5))

    return np.mean([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5], axis=0)

def precision_for_all(tar_pid_label, pred_pid_label, pred_pid_score):
    p_1 = []
    p_3 = []
    p_5 = []
    ndcg_1 = []
    ndcg_3 = []
    ndcg_5 = []
    for pid, score in pred_pid_score.items():
        indices = np.argsort(score)
        indices = indices[::-1]
        pre_labels = np.array(pred_pid_label[pid])[indices]
        true_labels = tar_pid_label[pid]
        r = [int(p_ in true_labels) for p_ in pre_labels]
        p_1.append(np.mean(r[:1]))
        p_3.append(np.mean(r[:3]))
        p_5.append(np.mean(r[:5]))
        ndcg_1.append(ndcg_at_k(r, 1))
        ndcg_3.append(ndcg_at_k(r, 3))
        ndcg_5.append(ndcg_at_k(r, 5))
    return np.mean([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5], axis=1)

def precision_for_pre_label(tar_pid_label, pred_pid_score):
    p_1 = []
    p_3 = []
    p_5 = []
    ndcg_1 = []
    ndcg_3 = []
    ndcg_5 = []
    for pid, true_labels in tar_pid_label.items():
        pre_labels = pred_pid_score[pid]
        r = [int(p_ in true_labels) for p_ in pre_labels]
        p_1.append(np.mean(r[:1]))
        p_3.append(np.mean(r[:3]))
        p_5.append(np.mean(r[:5]))
        ndcg_1.append(ndcg_at_k(r, 1))
        ndcg_3.append(ndcg_at_k(r, 3))
        ndcg_5.append(ndcg_at_k(r, 5))
    return np.mean([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5], axis=1)

def results_for_score_vector(tar_true_label_prop, tar_pid_y, pre_pid_score, pre_pid_prop):
    wts_p_1 = []
    wts_p_3 = []
    wts_p_5 = []
    wts_ndcg_1 = []
    wts_ndcg_3 = []
    wts_ndcg_5 = []
    for pid, y in tar_pid_y.items():
        true_label_prop = tar_true_label_prop[pid]
        pre_label_index = np.argsort(-np.array(pre_pid_score[pid]))[:5]
        # r = [y[ind] for ind in pre_label_index]
        # p_1.append(np.mean(r[:1]))
        # p_3.append(np.mean(r[:3]))
        # p_5.append(np.mean(r[:5]))
        # ndcg_1.append(ndcg_at_k(r, 1, len(true_label_prop)))
        # ndcg_3.append(ndcg_at_k(r, 3, len(true_label_prop)))
        # ndcg_5.append(ndcg_at_k(r, 5, len(true_label_prop)))
        # for propensity loss
        wts_r = [y[ind] * pre_pid_prop[pid][ind] for ind in pre_label_index]
        opt_r = sorted(true_label_prop, reverse=True)
        if len(opt_r) < 5:
            opt_r = opt_r + [0]*(5-len(opt_r))
        wts_p_1.append(np.mean(wts_r[:1]) / np.mean(opt_r[:1]))
        wts_p_3.append(np.mean(wts_r[:3]) / np.mean(opt_r[:3]))
        wts_p_5.append(np.mean(wts_r[:5]) / np.mean(opt_r[:5]))
        wts_ndcg_1.append(ndcg_at_k(wts_r, 1, 1) / ndcg_at_k(opt_r, 1, 1))
        wts_ndcg_3.append(ndcg_at_k(wts_r, 3, 3) / ndcg_at_k(opt_r, 3, 3))
        wts_ndcg_5.append(ndcg_at_k(wts_r, 5, 5) / ndcg_at_k(opt_r, 5, 5))
    return np.mean([wts_p_1, wts_p_3, wts_p_5, wts_ndcg_1, wts_ndcg_3, wts_ndcg_5], axis=1)

