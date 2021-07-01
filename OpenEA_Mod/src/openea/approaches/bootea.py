import gc
import math
import multiprocessing as mp
import random
import time
import numpy as np
import tensorflow as tf
import torch
from openea.modules.finding.alignment import compute_best_alignment_one_side, compute_bidirectional_alignments

from openea.modules.finding.evaluation import early_stop
import openea.modules.train.batch as bat
from openea.approaches.aligne import AlignE
from openea.modules.utils.util import task_divide
from openea.modules.bootstrapping.alignment_finder import find_potential_alignment_mwgm, check_new_alignment
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.load.kg import KG
from openea.modules.utils.util import load_session


# import openea.modules.load.read as rd <- Needed to save best embeddings


def bootstrapping(sim_mat, unaligned_entities1, unaligned_entities2, labeled_alignment, sim_th, k, len_valid_test):
    # Create alignments possible using the max-weight matching
    curr_labeled_alignment = find_potential_alignment_mwgm(sim_mat, sim_th, k, len_valid_test)
    if curr_labeled_alignment is not None:
        # Update labeled alignment: NOTE THIS IS DONE UNIDIRECTIONAL
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat,
                                                       len_valid_test)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat, len_valid_test)
        del curr_labeled_alignment
    if labeled_alignment is not None:
        # Save which entities from ref entities are now aligned to something
        # NOTE: we added the entities outside so this is not really "unaligned entities" anymore
        newly_aligned_entities1 = [unaligned_entities1[pair[0]] for pair in labeled_alignment]
        newly_aligned_entities2 = [unaligned_entities2[pair[1]] for pair in labeled_alignment]
    else:
        newly_aligned_entities1, newly_aligned_entities2 = None, None
    del sim_mat
    gc.collect()
    return labeled_alignment, newly_aligned_entities1, newly_aligned_entities2


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat, len_valid_test):
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i and i < len_valid_test:
            # If before it was matched right and now you're matching wrongly
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
            # Get previous match and check which was the similarity and new similarity
            pre_sim = sim_mat[i, pre_j]
            new_sim = sim_mat[i, j]
            if new_sim >= pre_sim:
                # If now you match it wrongly and before it was right and the similarity is increased
                if pre_j == i and j != i and i < len_valid_test:
                    n1 += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n1, "greedy update wrongly: ", n2)
    pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_new_alignment(pre_labeled_alignment, len_valid_test, context="after editing (<-)")
    return pre_labeled_alignment


def update_labeled_alignment_y(labeled_alignment, sim_mat, len_valid_test):
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        # Save all the i's which are aligned to j
        i_set = labeled_alignment_dict.get(j, set())
        i_set.add(i)
        labeled_alignment_dict[j] = i_set
    for j, i_set in labeled_alignment_dict.items():
        if len(i_set) == 1:
            # If j is aligned only with 1
            for i in i_set:
                updated_alignment.add((i, j))
        else:
            # Otherwise save as alignment the maximum similarity
            max_i = -1
            max_sim = -10
            for i in i_set:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_new_alignment(updated_alignment, len_valid_test, context="after editing (->)")
    return updated_alignment


def calculate_likelihood_mat(ref_ent1, ref_ent2, labeled_alignment):
    def set2dic(alignment):
        if alignment is None:
            return None
        dic = dict()
        for i, j in alignment:
            dic[i] = j
        assert len(dic) == len(alignment)
        return dic

    t = time.time()
    ref_mat = np.zeros((len(ref_ent1), len(ref_ent2)), dtype=np.float32)
    if labeled_alignment is not None:
        alignment_dic = set2dic(labeled_alignment)
        n = 1 / len(ref_ent1)
        for ii in range(len(ref_ent1)):
            if ii in alignment_dic.keys():
                ref_mat[ii, alignment_dic.get(ii)] = 1
            else:
                for jj in range(len(ref_ent1)):
                    ref_mat[ii, jj] = n
    print("calculate likelihood matrix costs {:.2f} s".format(time.time() - t))
    return ref_mat


def generate_supervised_triples(rt_dict1, hr_dict1, rt_dict2, hr_dict2, ents1, ents2):
    # Generate new triples of the format of training set (i.e. for each align, create the align also on the other KG)
    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_triples1.extend(generate_newly_triples(ents1[i], ents2[i], rt_dict1, hr_dict1))
        newly_triples2.extend(generate_newly_triples(ents2[i], ents1[i], rt_dict2, hr_dict2))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def mul(tensor1, tensor2, session, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = tf.matmul(tensor1, tensor2, transpose_b=True)
        if sigmoid:
            res = tf.sigmoid(sim_mat).eval(session=session)
        else:
            res = sim_mat.eval(session=session)
    else:
        res = np.matmul(tensor1.eval(session=session), tensor2.eval(session=session).T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


class BootEA(AlignE):

    def __init__(self):
        super().__init__()
        self.ref_ent1 = None
        self.ref_ent2 = None

    def init(self):
        # Defined in AlignE
        self._define_variables()
        self._define_embed_graph()
        # New
        self._define_alignment_graph()
        self._define_likelihood_graph()
        # Same as AlignE
        self.session = load_session(self.args.gpu)
        tf.global_variables_initializer().run(session=self.session)
        # Test shouldn't be here, we let BootEA overfits the datasets as the original authors does
        # THIS IS COMPLETELY WRONG, however we let BootEA having this big advantage and still performing worse
        # MAYBE IT'S NOT SO WRONG
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1 + self.kgs.extra_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2 + self.kgs.extra_entities2
        self.len_valid_test = len(self.kgs.valid_entities1 + self.kgs.test_entities1)

        # Added to improve early stopping -> Not needed anymore after rollback
        # self.saved_best = False
        # self.best_embeds = None

        # customize parameters
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

    def _define_alignment_graph(self):
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        phs = tf.nn.embedding_lookup(self.ent_embeds, self.new_h)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.new_r)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.new_t)
        # Define and optimize this alignment loss (may be Eq.8 in BootEA paper)
        self.alignment_loss = - tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))))
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)

    def _define_likelihood_graph(self):
        self.entities1 = tf.placeholder(tf.int32, shape=[None])
        self.entities2 = tf.placeholder(tf.int32, shape=[None])
        dim = len(self.kgs.valid_links) + len(self.kgs.test_entities1)
        dim1 = self.args.likelihood_slice
        self.likelihood_mat = tf.placeholder(tf.float32, shape=[dim1, dim])
        ent1_embed = tf.nn.embedding_lookup(self.ent_embeds, self.entities1)
        ent2_embed = tf.nn.embedding_lookup(self.ent_embeds, self.entities2)
        mat = tf.log(tf.sigmoid(tf.matmul(ent1_embed, ent2_embed, transpose_b=True)))
        # Define and optimize the likelihood loss (may be Eq.6 in BootEA paper)
        self.likelihood_loss = -tf.reduce_sum(tf.multiply(mat, self.likelihood_mat))
        self.likelihood_optimizer = generate_optimizer(self.likelihood_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def eval_ref_sim_mat(self):
        # return the similarity matrix from the embeddings
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2)
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1).eval(session=self.session)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1).eval(session=self.session)
        return np.matmul(refs1_embeddings, refs2_embeddings.T)

    def launch_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                              neighbors2):
        # iter over the iteration given (k)
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            # do one iteration (will use a batch, see internal definitions)
            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                             neighbors2)

    def train_alignment(self, kg1: KG, kg2: KG, entities1, entities2, training_epochs):
        if entities1 is None or len(entities1) == 0:
            return
        newly_tris1, newly_tris2 = generate_supervised_triples(kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict,
                                                               entities1, entities2)
        steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(training_epochs):
            t1 = time.time()
            alignment_loss = 0
            for step in range(steps):
                newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, self.args.batch_size)
                newly_batch1.extend(newly_batch2)
                alignment_fetches = {"loss": self.alignment_loss, "train_op": self.alignment_optimizer}
                alignment_feed_dict = {self.new_h: [tr[0] for tr in newly_batch1],
                                       self.new_r: [tr[1] for tr in newly_batch1],
                                       self.new_t: [tr[2] for tr in newly_batch1]}
                alignment_vals = self.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
                alignment_loss += alignment_vals["loss"]
            alignment_loss /= (len(newly_tris1) + len(newly_tris2))
            print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))

    def likelihood(self, labeled_alignment):
        # Looks like this is never used
        t = time.time()
        likelihood_mat = calculate_likelihood_mat(self.ref_ent1, self.ref_ent2, labeled_alignment)
        likelihood_fetches = {"likelihood_loss": self.likelihood_loss, "likelihood_op": self.likelihood_optimizer}
        likelihood_loss = 0.0
        steps = len(self.ref_ent1) // self.args.likelihood_slice
        ref_ent1_array = np.array(self.ref_ent1)
        ll = list(range(len(self.ref_ent1)))
        # print(steps)
        for i in range(steps):
            idx = random.sample(ll, self.args.likelihood_slice)
            likelihood_feed_dict = {self.entities1: ref_ent1_array[idx],
                                    self.entities2: self.ref_ent2,
                                    self.likelihood_mat: likelihood_mat[idx, :]}
            vals = self.session.run(fetches=likelihood_fetches, feed_dict=likelihood_feed_dict)
            likelihood_loss += vals["likelihood_loss"]
        print("likelihood_loss = {:.3f}, time = {:.3f} s".format(likelihood_loss, time.time() - t))

    def run(self):
        t = time.time()
        # Count total number of triples and how much step will need
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        # Decide which thread will get which step
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        # Initialize multiprocess architecture
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        # Initialize neighbors and how many iterations will have at max
        neighbors1, neighbors2 = None, None
        labeled_align = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            self.save(i)
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                       neighbors2)
            if i * sub_num >= self.args.start_valid:
                # validation using stop_metric (hits@1) -> used just to print their results
                self.valid(self.args.stop_metric)
                # Removed to be changed with our new validation procedure
                flag = self.valid_new_bootea(self.args.stop_metric_new, "bootea")
                # Check what we are doing on the test data, just for debug purpose
                # We are not using them to train
                # self.test_new()
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)

                # Code commented here used to save best embeddings -> Not used because we rolled back
                # to just keeping the last ones
                # self.flag1, new_flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                # if new_flag2 >= self.flag2:
                #     self.save_best_embeds()
                # self.flag2 = new_flag2

                if self.early_stop or i == iter_nums:
                    break
            labeled_align, entities1, entities2 = bootstrapping(self.eval_ref_sim_mat(),
                                                                self.ref_ent1, self.ref_ent2, labeled_align,
                                                                self.args.sim_th, self.args.k,
                                                                self.len_valid_test)
            # entities1, entities2 = self.bootstrapping_new()
            # Do an extra training trying to improve the matched embeddings
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)
            # self.likelihood(labeled_align)
            if i * sub_num >= self.args.start_valid:
                self.valid(self.args.stop_metric)
                self.valid_new_bootea(self.args.stop_metric_new, "bootea")
            t1 = time.time()
            assert 0.0 < self.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list1,
                                                 neighbors_num1, self.args.batch_threads_num)
            neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list2,
                                                 neighbors_num2, self.args.batch_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    def valid_new_bootea(self, stop_metric_new, method):
        # These two lists will map every index on the new generated embeddings to the original id it has
        # i.e. ids_2[0] = id of entity 0 from embeds2
        ids_1 = self.kgs.valid_entities1 + self.kgs.extra_entities_valid1
        ids_2 = self.kgs.valid_entities2 + self.kgs.extra_entities_valid2
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ids_1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ids_2).eval(session=self.session)
        return self.valid_new(stop_metric_new, method, ids_1, ids_2, embeds1, embeds2)

    def test_new(self):
        ids_1 = self.kgs.test_entities1 + self.kgs.valid_entities1 + self.kgs.extra_entities1
        ids_2 = self.kgs.test_entities2 + self.kgs.valid_entities2 + self.kgs.extra_entities2
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ids_1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ids_2).eval(session=self.session)
        self.test_new_common("bootea", ids_1, ids_2, embeds1, embeds2, self.args.csls)

        # CODE BELOW USED TO COMPARE RESULTS WITH THE BEST EMBEDDINGS SAVED VS USING THE LAST ONES
        # WE CHECKED THERE'S NO MUCH DIFFERENCE SO ROLLED BACK TO JUST USING LAST ONES
        # ids_1 = self.kgs.test_entities1 + self.kgs.valid_entities1 + self.kgs.extra_entities1
        # ids_2 = self.kgs.test_entities2 + self.kgs.valid_entities2 + self.kgs.extra_entities2
        # embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ids_1).eval(session=self.session)
        # embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ids_2).eval(session=self.session)
        # print("Results with the final embeddings")
        # self.test_new_common("bootea", ids_1, ids_2, embeds1, embeds2, self.args.csls)
        # embeds1 = self.best_embeds[ids_1]
        # embeds2 = self.best_embeds[ids_2]
        # print("Results with the best embeddings from early stop")
        # self.test_new_common("bootea", ids_1, ids_2, embeds1, embeds2, self.args.csls)

    # Our bootstrapping procedure -> unused because we rolled back to their bootstrapping
    def bootstrapping_new(self):
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1).eval(session=self.session)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2).eval(session=self.session)
        embeds1 = torch.from_numpy(refs1_embeddings).float().to(self.torch_device)
        embeds2 = torch.from_numpy(refs2_embeddings).float().to(self.torch_device)
        align_kg1_kg2 = compute_best_alignment_one_side(embeds1, embeds2, "bootea")
        align_kg2_kg1 = compute_best_alignment_one_side(embeds2, embeds1, "bootea")
        aligns = compute_bidirectional_alignments(align_kg1_kg2, align_kg2_kg1, self.ref_ent1, self.ref_ent2)
        print("Total aligned entities: ", len(aligns))
        if len(aligns) != 0:
            aligns1 = [e1 for (e1, e2) in aligns]
            aligns2 = [e2 for (e1, e2) in aligns]
        else:
            aligns1 = aligns2 = None
        return aligns1, aligns2

    # Functions to save best embeddings -> unused now because we just keep the last one
    # def save_best_embeds(self):
    #     self.best_embeds = self.ent_embeds.eval(session=self.session)
    #     self.saved_best = True
    #     print("Saved the best embeddings found so far!")
    # def save(self, i=None):
    #     if self.saved_best:
    #         ent_embeds = self.best_embeds
    #     else:
    #         ent_embeds = self.ent_embeds.eval(session=self.session)
    #     rel_embeds = self.rel_embeds.eval(session=self.session)
    #     mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
    #     rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat,
    #                        iteration=i)
