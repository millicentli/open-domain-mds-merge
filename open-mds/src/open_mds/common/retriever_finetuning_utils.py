"""
Utils for finetuning our retriever with our strategy.
"""

import copy
import random
import torch
import wandb

from open_mds.common import normalize_text
from sklearn import metrics

class RetrievalDataset(torch.utils.data.Dataset):
    """
    Retrieval dataset to return the positive, negatives, and queries
    """
    
    def __init__(
        self,
        input_data,
        n_random_negatives=1,
        normalize=False,
        training=False,
    ):
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self.data = input_data
        self.training = training
        self.n_random_negatives = n_random_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Nothing fancy, just gets the item for that row
        """
        example = self.data[index]
        question = example["question"]
        if self.training:
            # For training, we want to take a single sample and randomly select a gold example
            # From here, we want to sample some negatives -- maybe we can sample multiple negatives (n?)
            
            # Single gold/positive
            gold = random.choice(example["positive_ctxs"])
            # n > 0 negatives
            negatives = []
            random_negatives = random.sample(example["negative_ctxs"], self.n_random_negatives)
            negatives.extend(random_negatives)

            gold = gold["text"]
            negatives = [
                n["text"] for n in negatives
            ]
            
            example = {
                "query": self.normalize_fn(question),
                "gold": self.normalize_fn(gold),
                "negatives": [self.normalize_fn(n) for n in negatives],
            }

            return example
        else:
            # In the validation case, we want to return all possible outputs to evaluate
            # TODO: remove the queries -- only need qids for the golds and negs
            example = {
                "query": self.normalize_fn(question[0]),
                "qids": question[1],
                "gold": example["positive_ctxs"],
                "negatives": example["negative_ctxs"]
            }
            
            return example

class TrainCollator(object):
    """
    Define a collate function where we want to tokenize the following tokens:
    - q (query/text) tokens
    - p tokens
    - n tokens

    TODO: if possible, support:
    - In-batch negatives
    - Multiple negatives
    """

    def __init__(self, tokenizer, passage_maxlength=400):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [b['query'] for b in batch]
        positives = [b['gold'] for b in batch]
        negatives = [item for ex in batch for item in ex["negatives"]]

        qout = self.tokenizer.batch_encode_plus(
                queries,
                max_length=self.passage_maxlength,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
        )
        pout = self.tokenizer.batch_encode_plus(
            positives,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        nout = self.tokenizer.batch_encode_plus(
            negatives,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "q_tokens": qout["input_ids"],
            "q_mask": qout["attention_mask"],
            "p_tokens": pout["input_ids"],
            "p_mask": pout["attention_mask"],
            "n_tokens": nout["input_ids"],
            "n_mask": nout["attention_mask"]
        }

class EvalCollator(object):
    """
    For this collator function, we don't really care about the negatives and positives as embeds.
    Instead, we want to get all of the possible pmids and then convert to 0's and 1's for labels.

    Map these doc ids to 1s, 0s for each label.
    
    TODO: remove the queries -- only need qids for the golds and negs
    """

    def __init__(self, tokenizer, passage_maxlength=400):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [b['query'] for b in batch]
        qids = [b['qids'] for b in batch]

        gold_size = max([len(b['gold']) for b in batch])
        neg_size = max([len(b['negatives']) for b in batch])

        # Pad the positives, negatives
        gold_padded = []
        neg_padded = []
        for b in batch:
            gold_padded.append(b['gold'] + [None] * (gold_size - len(b['gold'])))
            neg_padded.append(b['negatives'] + [None] * (neg_size - len(b['negatives'])))

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "q_tokens": qout["input_ids"],
            "q_mask": qout["attention_mask"],
            "qids": qids,
            "gold": gold_padded,
            "negatives": neg_padded
        }

def mean_pooling(token_embeddings, mask):
    """
    Mean pooling is used to get the average embedding for Contriever.
    Other options include [CLS] pooling.
    """

    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def contrastive_loss(
    model,
    batch,
    temperature=0.7,
    label_smoothing=0.0
):
    """
    Contrastive loss, as defined in the Contriever paper.

    Takes in the passage, positive, negative, and tries to get the positives more
    similar to the passage than the negatives.

    Loss is InfoNCE loss, comes from: https://arxiv.org/abs/1807.03748
    """

    q_tokens, q_mask, p_tokens, p_mask, n_tokens, n_mask = batch.values()

    q_out = model(q_tokens, q_mask)
    p_out = model(p_tokens, p_mask)
    n_out = model(n_tokens, n_mask)

    q_embed = mean_pooling(q_out[0], q_mask)
    pn_embed = torch.cat((mean_pooling(p_out[0], p_mask), mean_pooling(n_out[0], n_mask)))

    scores = torch.einsum("id, jd->ij", q_embed / temperature, pn_embed)
    labels = torch.arange(len(q_embed), dtype=torch.long, device=q_embed.device)
    loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=label_smoothing)

    return loss

def calculate_auc(scores, qids, gold, neg):
    """
    Calculates ROC AUC given our inputs
    """
    auc_scores = []
    for qid, g, n in zip(qids, gold, neg):
        g_labels = [1 for i in g if i is not None]
        n_labels = [0 for i in n if i is not None]
        all_labels = g_labels + n_labels

        gold_scores = scores[(scores.qid == qid) & (scores.docno.isin(filter(lambda x: x != None, g)))]
        neg_scores = scores[(scores.qid == qid) & (scores.docno.isin(filter(lambda x: x != None, n)))]

        gold_scores = gold_scores['score'].tolist()
        neg_scores = neg_scores['score'].tolist()
        all_scores = gold_scores + neg_scores

        assert len(all_labels) == len(all_scores)

        roc_auc_score = metrics.roc_auc_score(all_labels, all_scores)
        auc_scores.append(roc_auc_score)

    return sum(auc_scores) / len(auc_scores)

def calculate_auprc(scores, qids, gold, neg):
    """
    Calculates AUPRC.
    """

    auprc_scores = []
    for qid, g, n in zip(qids, gold, neg):
        g_labels = [1 for i in g if i is not None]
        n_labels = [0 for i in n if i is not None]
        all_labels = g_labels + n_labels

        gold_scores = scores[(scores.qid == qid) & (scores.docno.isin(filter(lambda x: x != None, g)))]
        neg_scores = scores[(scores.qid == qid) & (scores.docno.isin(filter(lambda x: x != None, n)))]

        gold_scores = gold_scores['score'].tolist()
        neg_scores = neg_scores['score'].tolist()
        all_scores = gold_scores + neg_scores

        assert len(all_labels) == len(all_scores)

        auprc = metrics.average_precision_score(all_labels, all_scores)
        auprc_scores.append(auprc)

    return sum(auprc_scores) / len(auprc_scores)

def calculate_mrr(scores, qids, gold, neg):
    """
    Calculate MRR given we know our gold answers.
    The correct scores are the first scores.size()[-1] / 2 -- take those, get the MRR.

    The higher the MRR, the better.
    A score of 0 means that nothing was correct.
    """

    mrr_scores = []
    for qid, g, n in zip(qids, gold, neg):
        gold = list(filter(lambda x: x != None, g))
        neg = list(filter(lambda x: x != None, n))

        indices = list(scores[(scores.qid == qid) & (scores.docno.isin(g))]['rank'])

        size = len(indices)
        rr = sum(map(lambda x: 1 / (x + 1), indices))
        mrr_scores.append(rr / size)

    return sum(mrr_scores) / len(mrr_scores)

def transform_data(pt_dataset, n_random_negatives = 1, splits = ["train", "validation"]):
    """
    Takes as input the transformers dataset, extracts the query (question), and
    creates the negative and positive examples to be used.
    """

    hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)

    split_dict = {}
    for split in splits:
        topics = pt_dataset.get_topics(split)
        data = hf_dataset[split]
        pmids_to_abstracts = {p: a for pmid, abstract in zip(data['pmid'], data['abstract']) for p, a in zip(pmid, abstract)}
        all_pmids = list(set(pmids_to_abstracts.keys()))
        all_examples = []

        # Get all possible pmids and their corresponding data
        for idx, (query, data) in enumerate(zip(topics['query'], hf_dataset[split])):
            positive_pmids = data['pmid']
            subset_pmids = [pmid for pmid in all_pmids if pmid not in positive_pmids]
            if split == "train":
                negative_pmids = random.sample(subset_pmids, n_random_negatives)

                positive_ctxs = [{"text": pmids_to_abstracts[pmid]} for pmid in positive_pmids]
                negative_ctxs = [{"text": pmids_to_abstracts[pmid]} for pmid in negative_pmids]

                # Assertion to make sure we aren't double dipping
                assert positive_pmids[0] not in negative_pmids
                
                example = {}
                example['question'] = query
                example['positive_ctxs'] = positive_ctxs
                example['negative_ctxs'] = negative_ctxs
                all_examples.append(example)
            else:
                # If eval, we want to evaluate over the entire output dataset. So here,
                # we only really care about the pmids and their associated labels.
                example = {}
                example['question'] = (query, data['review_id'])
                example['positive_ctxs'] = positive_pmids
                example['negative_ctxs'] = subset_pmids
                all_examples.append(example)

        split_dict[split] = all_examples

    return split_dict