"""
Finetuning the retriever on the task-specific training data.

Datasets we train on:
MS^2

The format for these datasets is:
{
    "question": "What is the most popular operating system?", 
    "positive_ctxs": [{"text": "Windows is the most popular operating system."}], 
    "negative_ctxs": [{"text": "Windows is the most popular programming language."}],
    "hard_negative_ctxs": [{"text": "Windows is the most popular game console."}],
    "title": "Windows",
    "text": "Windows is the most popular operating system."
}

Here, we only use 'question', 'positive_ctxs', and 'negative_ctxs' in the training. This
information can be obtained by using the augmentations from the Open Domain MDS paper (Giorgi et al. 2023).

To use: make sure to do: export PYTHONPATH="${PYTHONPATH}:/home/li.mil/open-domain-mds-merge/open-mds/contriever"


To run the command:
python ./scripts/finetune_retriever.py "./conf_retriever/base.yml" "./conf_retriever/ms2/led-base/finetune.yml" \
    output_dir="./output/models/contriever" \

To put in scratch dir: /scratch/li.mil/open-domain-mds-merge/contriever

Plotting ROC Curve: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python


TODO: Add distributed training. Eval can simply be the accuracy and loss for now.
"""

import copy
import itertools
import logging
import numpy as np
import os
import random
import sys
import torch
import wandb

# from accelerate import Accelerator
from dataclasses import field, dataclass
from open_mds import indexing_basic
from open_mds.common import util, normalize_text
from rich import print
from sklearn.metrics import roc_auc_score
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm

from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    get_scheduler
)

from uncertainty_index_and_retrieve import Dataset

"""
Distributed training
"""

import torch.multiprocessing as mp

from src.open_mds.distributed_utils import setup, cleanup, prepare
from torch.nn.parallel import DistributedDataParallel as DDP

# Logging stuff
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Wandb stuff
wandb.login()

@dataclass
class RetrieverFinetuningArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    hf_dataset_name: Dataset = field(
        default="facebook/contriever",
        metadata={
            "case_sensitive": False,
            "help": "The name of a supported HuggingFace Dataset."
        }
    ),
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    ),
    output_dir: str = field(
        metadata={"help": "The output directory for the model and other output files."}
    ),
    seed: int = field(
        default=42,
        metadata={"help": "Seed to set for reproducibility."}
    )

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
        queries = [b['question'] for b in batch]
        # breakpoint()
        positives = [b['positive_ctxs']['text'] for b in batch]
        negatives = [b['negative_ctxs']['text'] for b in batch]

        # Old, batched with multiple inputs
        # positives = [b['text'] for item in batch for b in item['positive_ctxs']]
        # negatives = [b['text'] for item in batch for b in item['negative_ctxs']]

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
        queries = [b['question'] for b in batch]

        if len(batch) == 1:
            positives = [item['text'] for b in batch for item in b['positive_ctxs']]
            negatives = [item['text'] for b in batch for item in b['negative_ctxs']]
        else:
            raise NotImplementedError # Not sure if > 1 can be implemented

        # breakpoint()
        # positives = []
        # negatives = []
        # for b in batch:
        #     pos_items = [item['text'] for item in b['positive_ctxs']]
        #     neg_items = [item['text'] for item in b['negative_ctxs']]

        #     positives.append(pos_items)
        #     negatives.append(neg_items)

        # positives = [[item['text'] for item in b['positive_ctxs']] for b in batch]
        # negatives = [[item['text'] for item in b['negative_ctxs']] for b in batch]

        # Old, batched with multiple inputs
        # positives = [b['text'] for item in batch for b in item['positive_ctxs']]
        # negatives = [b['text'] for item in batch for b in item['negative_ctxs']]

        # Find the number of possible outputs for positives, negatives
        # max_pos = max([len(positive) for positive in positives])
        # max_neg = max([len(negative) for negative in negatives])
        # max_inner_pos = max([len(p) for pos in positives for p in pos])
        # max_inner_neg = max([len(n) for neg in negatives for n in neg])

        # Next, concatenate the positives, negatives and then tokenize/encode
        # Want to go from batch size x no. subsamples x seq length to
        # batch size x no subsamples * seq length

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # pos_list = []
        # neg_list = []
        # for pos, neg in zip(positives, negatives):
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
        # pos_list.append(pout)
            # neg_list.append(nout)

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
    # breakpoint()
    q_embed = mean_pooling(q_out[0], q_mask)
    pn_embed = torch.cat((mean_pooling(p_out[0], p_mask), mean_pooling(n_out[0], n_mask)))
    scores = torch.einsum("id, jd->ij", q_embed / temperature, pn_embed)
    labels = torch.arange(len(q_embed), dtype=torch.long, device=q_embed.device)

    loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=label_smoothing)
    
    argmax = torch.argmax(scores, dim=1)

    # print(f"Debugging: ", argmax)
    # breakpoint()
    return loss

def calculate_auc(scores):
    """
    Calculates AUC given our inputs.

    First, take our input scores, see if it's in the correct side (1 if correct, 0 otherwise)
    """
    values, indices = torch.topk(scores, scores.shape[-1], dim=1)

    half = int(scores.shape[-1] / 2)
    # pos_correctness = [1 if indices[0][idx] < half else 0 for idx in range(half)]
    # neg_correctness = [1 if indices[0][idx + half] >= half else 0 for idx in range(half)]
    # all_correctness = pos_correctness + neg_correctness

    target_correctness = [1] * half + [0] * half

    outputs = torch.nn.functional.sigmoid(scores)

    # assert len(all_correctness) == len(target_correctness)
    breakpoint()
    roc_scores = roc_auc_score(target_correctness, outputs.squeeze().cpu().numpy().tolist())

    breakpoint()
    print("hi")


@torch.no_grad()
def calculate_metrics(
    model,
    batch,
    debug=True,
    mrr=True,
    auc=True
):
    """
    Calculates accuracy based on the criteria, whether that be:
    - Mean Reciprocal Rank (MRR), or
    - Area Under Curve (AUC)

    Includes some debugging in this part if needed

    TODO: decide if we apply temperature here or not...
    """

    q_tokens, q_mask, p_tokens, p_mask, n_tokens, n_mask = batch.values()

    q_out = model(q_tokens, q_mask)
    p_out = model(p_tokens, p_mask)
    n_out = model(n_tokens, n_mask)
    # breakpoint()
    q_embed = mean_pooling(q_out[0], q_mask)
    pn_embed = torch.cat((mean_pooling(p_out[0], p_mask), mean_pooling(n_out[0], n_mask)))
    scores = torch.einsum("id, jd->ij", q_embed, pn_embed)
    # breakpoint()
    # if debug:
    labels = torch.arange(len(q_embed), dtype=torch.long, device=q_embed.device)
    loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=0.0)
    # print(f"Loss: {loss.item()}")

    wandb.log({"Validation loss": loss.item()})

    # argmax = torch.argmax(scores, dim=1)
    
    if auc:
        # Calculate AUC
        calculate_auc(scores)
    if mrr:
        # Calculate MRR
        raise NotImplementedError

    # breakpoint()
    # print(f"Debugging eval:", argmax)


def finetune(
    model,
    tokenizer,
    # accelerator,
    output_dir,
    train_dataset,
    eval_dataset,
    lr=0.05,
    num_epochs=100,
    train_batch_size=16,
    # eval_batch_size=16,
    eval_batch_size=1, # TODO: figure out if it's batchable. Don't think it is because OOM
    log_freq=1,
):
    # Enable optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # train_dataset = train_dataset[:1000]
    # Enable sampler
    train_sampler = RandomSampler(train_dataset)
    # Enable collator, dataloader
    collator = TrainCollator(tokenizer, passage_maxlength=512)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collator
    )
    
    # Enable scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    logger.info(f" Num training steps: {num_training_steps}")

    # train_dataloader, model, optim = accelerator.prepare(
    #     train_dataloader, model, optim
    # )

    # Start train loop
    model.train()
    model.cuda()

    step = 0
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader, desc=f"Training, Epoch {epoch}"):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            # breakpoint()
            loss = contrastive_loss(
                model,
                batch,
                temperature=0.7,
                label_smoothing=0.0
            )

            loss.backward()
            # accelerator.backward(loss)

            lr_scheduler.step()
            optim.step()
            optim.zero_grad()
            # breakpoint()
            step += 1
        
            if step % log_freq == 0:
                log = f"{step} / {num_training_steps}"
                
                log += f" | lr: {lr_scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                log += f" | Loss: {loss.item()}"
                logger.info(log)

                wandb.log({"Training loss": loss.item()})

                evaluate(
                    model,
                    tokenizer,
                    # accelerator,
                    eval_dataset,
                    eval_batch_size=eval_batch_size
                )

                model.save_pretrained(f"{output_dir}/model_{step}", from_pt=True) 

        
@torch.no_grad()
def evaluate(model, tokenizer, eval_dataset, eval_batch_size):
    eval_sampler = SequentialSampler(eval_dataset)

    # Enable collator, dataloader
    collator = EvalCollator(tokenizer, passage_maxlength=512)
    # collator = TrainCollator(tokenizer, passage_maxlength=512)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        collate_fn=collator
    )

    # eval_dataloader = accelerator.prepare(
    #     eval_dataloader
    # )

    model.eval()

    # breakpoint()
    for batch in tqdm(eval_dataloader, desc=f"Evaluation", total=len(eval_dataloader)):
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        calculate_metrics(
            model,
            batch,
            auc=False,
            mrr=False
        )



def transform_data(pt_dataset, splits = ["train", "validation"]):
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
            negative_pmids = random.sample(subset_pmids, len(data['pmid']))

            positive_ctxs = [{"text": pmids_to_abstracts[pmid]} for pmid in positive_pmids]
            negative_ctxs = [{"text": pmids_to_abstracts[pmid]} for pmid in negative_pmids]

            # Assertion to make sure we aren't double dipping
            assert positive_pmids[0] not in negative_pmids
            
            example = {}
            if split == "train":
                # Choose a single gold context
                # Then, sample a bunch of negatives -- we'll sample a single one here
                # TODO: add options to increase the number of negatives
                # pos = random.choice(positive_ctxs)
                # neg = random.choice(negative_ctxs)

                # example['question'] = query
                # example['positive_ctxs'] = pos
                # example['negative_ctxs'] = neg
                # all_examples.append(example)

                # OLD STUFF BELOW
                # # TODO: Need to make sure to scramble the inputs when the model makes the predictions, right now they're all in the same order
                cartesians = itertools.product(positive_ctxs, negative_ctxs)
                cartesians_list = list(cartesians)

                for i, (pos, neg) in enumerate(cartesians_list):
                    example = {}
                    example['question'] = query
                    example['positive_ctxs'] = pos
                    example['negative_ctxs'] = neg

                    all_examples.append(example)
            else:
                # Split is everything else, so validation or test
                # Here, for each sample, we're going to reconstruct the dataset to include all positives and negatives together
                example['question'] = query
                example['positive_ctxs'] = positive_ctxs
                example['negative_ctxs'] = negative_ctxs
                
                all_examples.append(example)

        random.shuffle(all_examples)
        split_dict[split] = all_examples

    return split_dict

class RetrievalDataset(torch.utils.data.Dataset):
    """
    Retrieval dataset to return the positive, negatives, and queries
    """
    
    def __init__(
        self,
        input_data,
        normalize=False,
        training=False
    ):
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Nothing fancy, just gets the item for that row
        """
        example = self.data[index]

        if self.training:
            gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]
            negative = n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"]
            
            example = {
                "query": self.normalize_fn(question),
                "gold": self.normalize_fn(gold),
                "negatives": [self.normalize_fn(n) for n in negatives],
            }

            return example
        else:
            breakpoint()
            raise NotImplementedError
            
def main():
    """Trains the retriever on the current dataset"""
    parser = HfArgumentParser(
        (RetrieverFinetuningArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    # Our unique parsing strategy (which depends on OmegaConf) exists here
    elif any(argv.endswith(".yml") for argv in sys.argv[1:]):
        conf = util.parse_omega_conf()
        retriever_args = parser.parse_dict(conf)[0]
    else:
        retriever_args = parser.parse_args_into_dataclasses()[0]

    if retriever_args.hf_dataset_name == Dataset.ms2:
        pt_dataset = indexing_basic.MSLR2022Dataset(name=retriever_args.hf_dataset_name)
    else:
        raise NotImplementedError

    # start a new wandb run to track this script
    # TODO: integrate wandb with DDP: https://docs.wandb.ai/guides/track/log/distributed-training
    wandb.init(
        # set the wandb project where this run will be logged
        project="retriever-finetuning-ms2",

        # track hyperparameters and run metadata
        config={
            "architecture": "Contriever",
            "dataset": "ms2"
        }
        # config={
        # "learning_rate": 0.05,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        # "epochs": 10,
        # }
    )

    # suppose we have 3 gpus
    world_size = 3    
    mp.spawn(
        start,
        args=(world_size),
        nprocs=world_size
    )

def start(rank, world_size):
    # Init distributed training
    setup(rank, world_size)

    print(
        f"[bold]:book: Dataset chosen: '{retriever_args.hf_dataset_name}'... [/bold]"
    )
    
    set_seed(retriever_args.seed)

    model = AutoModel.from_pretrained(retriever_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(retriever_args.model_name_or_path)

    # Put model on DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    splits = ["train", "validation"]
    dataset_dict = transform_data(pt_dataset, splits)

    # Initialize everything for training
    logger.info("Start training")
    finetune(
        model,
        tokenizer,
        # accelerator,
        retriever_args.output_dir,
        dataset_dict['train'],
        dataset_dict['validation'],
    )

    # Cleans up all of the distributed processes
    cleanup()


if __name__ == "__main__":
    main()