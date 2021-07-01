import numpy as np

from openea.modules.finding.alignment import greedy_alignment


def valid(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False):
    if mapping is None:
        # call the greedy alignment (used in BootEA)
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                      metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                      metric, normalize, csls_k, accurate)
    return hits1_12, mrr_12


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate)
    return alignment_rest_12, hits1_12, mrr_12


def early_stop(flag1, flag2, flag):
    # if we reached the value we wanted in the metric we can stop
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


'''New evaluation procedure functions using the new alignments (By the Entity-Matchers!)'''


def compute_prec_rec_f1(aligns, truth_links):
    aligns = set(aligns)
    truth_links = set(truth_links)
    num_correct = len(aligns.intersection(truth_links))
    if num_correct == 0 or len(aligns) == 0:
        print("Got 0, 0, 0 in evaluation!!!")
        return 0, 0, 0
    precision = num_correct / len(aligns)
    recall = num_correct / len(truth_links)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
