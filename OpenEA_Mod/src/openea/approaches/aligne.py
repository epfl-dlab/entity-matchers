import tensorflow as tf

from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import limited_loss
from openea.models.basic_model import BasicModel


class AlignE(BasicModel):

    def __init__(self):
        super().__init__()
        self.rel_ids, self.func_val = [], []

    def init(self):
        # Get functionalities in the format required by TF
        rel_ids1, func_val1 = format_func(self.kgs.rel_func1)
        rel_ids2, func_val2 = format_func(self.kgs.rel_func2)
        self.rel_ids = rel_ids1 + rel_ids2
        self.func_val = func_val1 + func_val2

        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)
        tf.tables_initializer().run(session=self.session)

        # customize parameters -> note they are the same as in BootEA json
        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'swapping'
        assert self.args.loss == 'limited'
        assert self.args.neg_sampling == 'truncated'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.pos_margin >= 0.0
        assert self.args.neg_margin > self.args.pos_margin

        assert self.args.neg_triple_num > 1
        assert self.args.truncated_epsilon > 0.0
        assert self.args.learning_rate >= 0.01

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            # initialize embeddings for entity and relations (BootEA -> normal distribution l2 normalized)
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)
            self.func_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(tf.constant(self.rel_ids, dtype=tf.int64),
                                                    tf.constant(self.func_val)), -1)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            # create placeholders for all positive and negative samples?? -> head, relation, tail
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs_func = tf.placeholder(tf.int64, shape=[None])
            self.neg_rs_func = tf.placeholder(tf.int64, shape=[None])
        with tf.name_scope('triple_lookup'):
            # get embeddings from the initialized random variables,
            # should given a listen of id but we give just none here, don't quite understood it
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)
        with tf.name_scope('functionalities_lookup'):
            pos_func = self.func_lookup.lookup(self.pos_rs_func)
            neg_func = self.func_lookup.lookup(self.neg_rs_func)
        with tf.name_scope('triple_loss'):
            # Compute loss and use optimizer (Adagrad for BootEA) to apply gradients
            self.triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts, pos_func, neg_func,
                                            self.args.pos_margin, self.args.neg_margin,
                                            self.args.loss_norm, use_func=self.args.use_func,
                                            balance=self.args.neg_margin_balance)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)


def format_func(rel_func: dict):
    rel_ids = list(rel_func.keys())
    func_vals = list(rel_func.values())
    return rel_ids, func_vals
