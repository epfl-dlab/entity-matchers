import sys
sys.path.append('../')
import deepmatcher as dm
import argparse
import os
import pandas as pd
import time

parser = argparse.ArgumentParser(
    description="run experiment deepmatchers"
)

parser.add_argument(
    "--dataset_deepmatcher",
    type=str,
    help=''
)
parser.add_argument(
    "--dataset_original",
    type=str,
    help=''
)
parser.add_argument(
    "--folds",
    type=int,
    nargs='+',
    help='list of folds numbers',
    required=True
)
args_main = parser.parse_args()

if __name__ == "__main__":
    EPOCHS = 20
    folds = args_main.folds
    for fold in folds:
        test_data = []
        with open("{}/721_5folds/{}/test_links".format(args_main.dataset_original, fold)) as f:
            for l in f:
                test_data.append(l.rstrip("\n").split("\t"))
        print("The original test data has length: {}".format(len(test_data)))
        df_train = pd.read_csv("{}/721_5folds/{}/train.csv".format(args_main.dataset_deepmatcher, fold))
        pos_neg_ratio = int((len(df_train) - df_train['label'].sum()) / df_train['label'].sum())
        pos_neg_ratio = pos_neg_ratio if pos_neg_ratio > 0 else 1
        print("The positive negative ratio is {}".format(pos_neg_ratio))
        # Get data
        train, validation, test = dm.data.process(path="{}/721_5folds/{}".format(args_main.dataset_deepmatcher, fold),
            train='train.csv', validation='valid.csv', test='test.csv', left_prefix='ltable_', right_prefix='rtable_',
                                                  ignore_columns=['rtable_id', 'ltable_id'])
        # Initialize the model
        model = dm.MatchingModel()

        # train
        start = time.time()
        model.run_train(train, validation,
                        best_save_path='best_model_{}_{}.pth'.format("_".join(args_main.dataset_deepmatcher.split("/")[-2:]), fold),
                        epochs=EPOCHS,
                        pos_neg_ratio=pos_neg_ratio)
        time_train = time.time() - start
        print("Total time for train: {}".format(time_train))

        # test
        start = time.time()
        print("Eval test:\n")
        tp1, tn1, fp1, _ = model.run_eval(test, return_predictions=True, device='cpu')
        fn1 = len(test_data) - tp1
        print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp1, tn1, fp1, fn1))
        prec1 = tp1 / (tp1 + fp1)
        rec1 = tp1 / (tp1 + fn1)
        print("prec: {}, recall: {}, f1: {}".format(
            prec1, rec1, 2 * prec1 * rec1 / (prec1 + rec1)
        ))
        eval_time = time.time() - start
        print("total time for eval: {}".format(eval_time))


