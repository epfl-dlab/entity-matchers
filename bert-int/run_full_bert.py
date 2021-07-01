import argparse
import time

from basic_bert_unit.main import main as main_basic
from interaction_model.clean_attribute_data import main as main_clean_attrs
from interaction_model.get_entity_embedding import main as main_get_embeds
from interaction_model.get_attributeValue_embedding import main as main_get_attr_embeds
from interaction_model.get_neighView_and_desView_interaction_feature import main as main_get_neigh_embeds
from interaction_model.get_attributeView_interaction_feature import main as main_get_attr_view_embeds
from interaction_model.interaction_model import main as main_interaction_model


def align(best_pairs_1_2, best_pairs_2_1):
    aligns = []
    pairs_2_1_dict = dict(best_pairs_2_1)
    for (e1, e2) in best_pairs_1_2:
        if e2 not in pairs_2_1_dict:
            continue
        elif pairs_2_1_dict[e2] == e1:
            aligns.append((e1, e2))
    return aligns


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


def read_truth(path):
    truth = []
    with open(path) as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            truth.append((e1, e2))
    return truth


def run_bert(args):
    start_time = time.time()
    # Run trying to align KG1 to KG2
    main_basic(args)
    main_clean_attrs(args)
    main_get_embeds(args)
    main_get_attr_embeds(args)
    main_get_neigh_embeds(args)
    main_get_attr_view_embeds(args)
    best_pairs_1_2 = main_interaction_model(args)

    # Run KG2 to KG1
    main_basic(args, swap=True)
    main_clean_attrs(args, swap=True)
    main_get_embeds(args)
    main_get_attr_embeds(args)
    main_get_neigh_embeds(args)
    main_get_attr_view_embeds(args)
    best_pairs_2_1 = main_interaction_model(args)
    print("Training ends. Total time = {:.3f} s.".format(time.time() - start_time))

    best_pairs_test = align(best_pairs_1_2, best_pairs_2_1)
    precision, recall, f1 = compute_prec_rec_f1(best_pairs_test,
                                                read_truth(args.dataset + "/" + args.fold_folder + "/test_links"))
    print("Final test result:")
    print("\tPrecision:", precision)
    print("\tRecall:", recall)
    print("\tF1:", f1)
    print("Total run time = {:.3f} s.".format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BERT-INT')
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--fold_folder", type=str, help="Folder to use")
    parser.add_argument("--gpu", type=int, help="GPU to use", default=0)
    parser.add_argument("--dict_path", type=str, default=None, help="Path to abstract dict")

    parser.add_argument("--margin_basic", type=float, default=3, help="Margin for basic bert unit")
    parser.add_argument("--lr_basic", type=float, default=1e-5, help="Learning rate for basic bert unit")
    parser.add_argument("--train_batch_size_basic", type=int, default=24, help="Train batch size for basic bert unit")
    parser.add_argument("--test_batch_size_basic", type=int, default=128, help="Test batch size for basic bert unit")
    parser.add_argument("--neg_num_basic", type=int, default=2, help="Sample negative number for basic bert unit")

    parser.add_argument("--margin_interaction", type=float, default=1, help="Margin for interaction model")
    parser.add_argument("--lr_interaction", type=float, default=5e-4, help="Learning rate for interaction model")
    parser.add_argument("--train_batch_size_interaction", type=int, default=128,
                        help="Train batch size for interaction model")
    parser.add_argument("--neg_num_interaction", type=int, default=5,
                        help="Sample negative number for interaction model")

    args = parser.parse_args()
    run_bert(args)
