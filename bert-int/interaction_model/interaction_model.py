from .model_train_test_func import *
from .Param import *
from .utils import fixed


def main(args, swap=False):
    print("----------------interaction model--------------------")
    cuda_num = args.gpu
    print("GPU num {}".format(cuda_num))
    # print("ko~ko~da~yo~")
    dataset_prefix = args.dataset.split("/")[-1]
    print("Dataset prefix:", dataset_prefix)
    if args is not None:
        save_read_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + dataset_prefix + 'save_read_data.pkl'
        _, _, index2entity, _, _, _, _ = pickle.load(open(save_read_data_path, "rb"))

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + dataset_prefix + 'other_data.pkl'
    train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))
    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill), len(test_ill), len(
        set(train_ill) & set(test_ill))))

    # (candidate) entity pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH.format(dataset_prefix), "rb"))

    # interaction features
    nei_features = pickle.load(
        open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH.format(dataset_prefix), "rb"))  # neighbor-view interaction similarity feature
    att_features = pickle.load(
        open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH.format(dataset_prefix), 'rb'))  # attribute-view interaction similarity feature
    des_features = pickle.load(
        open(DESVIEW_SIMILARITY_FEATURE_PATH.format(dataset_prefix), "rb"))  # description/name-view interaction similarity feature
    train_candidate = pickle.load(open(TRAIN_CANDIDATES_PATH.format(dataset_prefix), "rb"))
    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH.format(dataset_prefix), "rb"))
    all_features = []  # [nei-view cat att-view cat des/name-view]
    for i in range(len(entity_pairs)):
        all_features.append(nei_features[i] + att_features[i] + des_features[i])  # 42 concat 42 concat 1.
    print("All features embedding shape: ", np.array(all_features).shape)

    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    Train_gene = Train_index_generator(train_ill, train_candidate, entpair2f_idx, neg_num=args.neg_num_interaction,
                                       batch_size=args.train_batch_size_interaction)
    Model = MlP(42 * 2 + 1, 11).cuda(cuda_num)
    Optimizer = optim.Adam(Model.parameters(), lr=args.lr_interaction)
    Criterion = nn.MarginRankingLoss(margin=args.margin_interaction, size_average=True)

    save_out_ent_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + dataset_prefix + 'save_ent_out.pkl'
    ent_out_1, _ = pickle.load(open(save_out_ent_path, "rb"))

    # train
    best_pairs = train(Model, Optimizer, Criterion, Train_gene, all_features, test_candidate, test_ill,
          entpair2f_idx, ent_out_1, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=50)

    # save
    if not swap:
        torch.save(Model, open(INTERACTION_MODEL_SAVE_PATH.format(dataset_prefix, "1_to_2"), "wb"))
    else:
        torch.save(Model, open(INTERACTION_MODEL_SAVE_PATH.format(dataset_prefix, "2_to_1"), "wb"))
    best_pairs_entities = [(index2entity[e1], index2entity[e2]) for (e1, e2) in best_pairs]
    return best_pairs_entities


if __name__ == '__main__':
    fixed(SEED_NUM)
    main(None)
