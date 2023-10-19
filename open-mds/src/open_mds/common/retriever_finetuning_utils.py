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
            example = {
                "query": self.normalize_fn(question),
                "gold": [self.normalize_fn(gold["text"]) for gold in example["positive_ctxs"]],
                "negatives": [self.normalize_fn(n["text"]) for n in example["negative_ctxs"]]
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
    Define a collate function where we want to tokenize the following tokens:
    - q (query/text) tokens
    - p tokens
    - n tokens

    For eval, our batch is the set of documents related to the question (and the
    associated negatives) per sample.

    TODO: if possible, support:
    - In-batch negatives
    - Multiple negatives
    """

    def __init__(self, tokenizer, passage_maxlength=400):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [b['query'] for b in batch]
        if len(batch) == 1:
            positives = [item for item in batch[0]["gold"]]
            negatives = [item for item in batch[0]["negatives"]]
        else:
            raise NotImplementedError # Not sure if > 1 can be implemented

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

def calculate_auc(scores):
    """
    Calculates AUC given our inputs.

    First, take our input scores, see if it's in the correct side (1 if correct, 0 otherwise)
    """
    labels = torch.zeros_like(scores)
    size = int(scores.size()[-1] / 2)
    labels[:, :size] = 1

    values, indices = torch.topk(scores, scores.shape[-1], dim=1)
    outputs = torch.nn.functional.sigmoid(scores)

    return metrics.roc_curve(labels.squeeze().cpu().tolist(), outputs.squeeze().cpu().numpy().tolist())

def calculate_mrr(scores):
    """
    Calculate MRR given we know our gold answers.
    The correct scores are the first scores.size()[-1] / 2 -- take those, get the MRR.

    The higher the MRR, the better.
    A score of 0 means that nothing was correct.
    """
    labels = torch.zeros_like(scores)
    size = int(scores.size()[-1] / 2)
    labels[:, :size] = 1

    sorted_scores, indices = torch.sort(scores, descending=True)

    correct_mask = (indices < size).squeeze()
    correct_indices = torch.arange(indices.size()[-1], device=indices.device)[correct_mask]

    rr = sum(map(lambda x: 1 / (x.item() + 1), correct_indices))

    return rr / size

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
            # print("Here'x idx:", idx)
            positive_pmids = data['pmid']
            subset_pmids = [pmid for pmid in all_pmids if pmid not in positive_pmids]
            if split == "train":
                negative_pmids = random.sample(subset_pmids, n_random_negatives)
            else:
                # If eval, our neg_pmids are dependent on size of train
                negative_pmids = random.sample(subset_pmids, len(positive_pmids))

            positive_ctxs = [{"text": pmids_to_abstracts[pmid]} for pmid in positive_pmids]
            negative_ctxs = [{"text": pmids_to_abstracts[pmid]} for pmid in negative_pmids]

            # Assertion to make sure we aren't double dipping
            assert positive_pmids[0] not in negative_pmids
            
            example = {}
            example['question'] = query
            example['positive_ctxs'] = positive_ctxs
            example['negative_ctxs'] = negative_ctxs
            all_examples.append(example)

        split_dict[split] = all_examples

    return split_dict