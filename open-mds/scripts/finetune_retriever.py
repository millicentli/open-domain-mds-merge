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

python ./scripts/finetune_retriever.py "./conf_retriever/base.yml" "./conf_retriever/ms2/led-base/finetune.yml" \
    output_dir="/scratch/li.mil/open-domain-mds-merge/contriever" \

To put in scratch dir: /scratch/li.mil/open-domain-mds-merge/contriever

Plotting ROC Curve: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python


TODO: Add distributed training. Eval can simply be the accuracy and loss for now.
"""

import copy
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import wandb

from dataclasses import field, dataclass
from open_mds import indexing_basic
from open_mds.common import util
from pathlib import Path
from platformdirs import user_cache_dir
from rich import print
# from sklearn.metrics import roc_auc_score
from sklearn import metrics
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm

from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    get_scheduler,
    TrainingArguments
)

from uncertainty_index_and_retrieve import Dataset

from src.open_mds.common.retriever_finetuning_utils import (
    TrainCollator,
    EvalCollator,
    mean_pooling,
    contrastive_loss,
    calculate_auc,
    calculate_mrr,
    transform_data,
    RetrievalDataset
)

"""
Distributed training with DP
"""

from torch.nn.parallel import DataParallel

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
        metadata={
            "case_sensitive": False,
            "help": "The name of a supported HuggingFace Dataset."
        }
    ),
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    ),
    n_random_negatives: int = field(
        default=2,
        metadata={"help": "Number of random negatives to sample."}
    )
    relevance_cutoff: int = field(
        default=100,
        metadata={"help": "The relevance cutoff that we'll use."}
    )
    log_freq: int = field(
        default=1000,
        metadata={"help": "How often to log output during training."}
    )

def finetune(
    model,
    tokenizer,
    output_dir,
    pt_dataset,
    n_random_negatives,
    lr=1e-4,
    num_epochs=1,
    train_batch_size=32,
    eval_batch_size=1, # TODO: figure out if it's batchable. Don't think it is because OOM
    weight_decay=0.01,
    log_freq=1000,
    relevance_cutoff=100
):
    pt_dataset_do_train = copy.deepcopy(pt_dataset)
    del pt_dataset_do_train._hf_dataset['train']
    del pt_dataset_do_train._hf_dataset['test']

    # Make the splits
    splits = ["train", "validation"]
    dataset_dict = transform_data(
        pt_dataset,
        n_random_negatives=n_random_negatives,
        splits=splits
    )
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"]

    # Enable optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = RetrievalDataset(train_dataset, training=True, n_random_negatives=n_random_negatives)
    
    # Enable sampler
    train_sampler = RandomSampler(train_dataset)

    # Enable collator, dataloader
    collator = TrainCollator(tokenizer, passage_maxlength=512)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collator,
        num_workers=1
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

    # Start train loop
    model.train()
    model.cuda()

    step = 0
    max_recall = -1000
    max_precision = -1000
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader, desc=f"Training, Epoch {epoch}"):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            loss = contrastive_loss(
                model,
                batch,
                temperature=0.7,
                label_smoothing=0.0
            )

            loss.backward()

            lr_scheduler.step()
            optim.step()
            optim.zero_grad()

            step += 1
        
            if step % log_freq == 0:
                log = f"{step} / {num_training_steps}"
                
                log += f" | lr: {lr_scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                log += f" | Loss: {loss.item()}"
                logger.info(log)

                wandb.log({"Training loss": loss.item(), "Learning rate": lr_scheduler.get_lr()[0]})

                # Save the model first here
                model.module.save_pretrained(f"{output_dir}/model_{step}", from_pt=True)
                tokenizer.save_pretrained(f"{output_dir}/model_{step}")

                metrics = pt_evaluation(
                    f"{output_dir}/model_{step}",
                    pt_dataset_do_train,
                    output_dir,
                    relevance_cutoff=relevance_cutoff,
                    split="validation"
                )

                print(metrics)

                metrics = metrics.values[0]
                wandb.log({
                    "P@1": metrics[1],
                    "R@1": metrics[2],
                    "P@5": metrics[3],
                    "R@5": metrics[4],
                    f"P@{relevance_cutoff}": metrics[5],
                    f"R@{relevance_cutoff}": metrics[6]
                })

                # We save a model depending on the P@{relevance_cutoff} value, call it best_model
                r_cutoff = metrics[-1]
                p_cutoff = metrics[-2]

                if r_cutoff > max_recall and p_cutoff > max_precision:
                    max_recall = r_cutoff
                    max_precision = p_cutoff

                    # If it's the best model, save it as the best model
                    model.module.save_pretrained(f"{output_dir}/model_best", from_pt=True)
                    tokenizer.save_pretrained(f"{output_dir}/model_best")

    # Save the very last model anyways, for posterity
    model.module.save_pretrained(f"{output_dir}/model_{step}_last", from_pt=True)
    tokenizer.save_pretrained(f"{output_dir}/model_{step}_last")

    # Do some final evaluation on the validation set
    auc, mrr = evaluate(model, tokenizer, eval_dataset, eval_batch_size=eval_batch_size)
    mrr = sum(mrr) / len(mrr)
    
    # Plot this final evaluation
    plt.title("AUC Curve over the validation dataset")
    plt.ylabel("True Positives Rate")
    plt.xlabel("False Positive Rate")
    for datapoint in auc:
        fpr, tpr, _ = datapoint
        plt.plot(fpr, tpr, alpha=0.7)
    wandb.log({'AUC Curve over validation': plt})

    logger.info(f" Average MRR over validation dataset: {mrr}")
        
@torch.no_grad()
def evaluate(model, tokenizer, eval_dataset, eval_batch_size=1):
    breakpoint()
    eval_dataset = RetrievalDataset(eval_dataset)
    eval_sampler = SequentialSampler(eval_dataset)

    # Enable collator, dataloader
    collator = EvalCollator(tokenizer, passage_maxlength=512)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        collate_fn=collator
    )

    auc_scores = []
    mrr_scores = []
    model.eval()
    for batch in tqdm(eval_dataloader, desc=f"Evaluation", total=len(eval_dataloader)):
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        q_tokens, q_mask, p_tokens, p_mask, n_tokens, n_mask = batch.values()

        p_size = len(p_tokens)
        assert p_size == len(n_tokens)

        q_out = model(q_tokens, q_mask)
        p_out = model(p_tokens, p_mask)
        n_out = model(n_tokens, n_mask)

        q_embed = mean_pooling(q_out[0], q_mask)
        pn_embed = torch.cat((mean_pooling(p_out[0], p_mask), mean_pooling(n_out[0], n_mask)))
        scores = torch.einsum("id, jd->ij", q_embed, pn_embed)

        # Calculate ROC auc averages
        auc = calculate_auc(scores)
        auc_scores.append(auc)

        # Calculate MRR
        mrr = calculate_mrr(scores)
        mrr_scores.append(mrr)

    return auc_scores, mrr_scores

@torch.no_grad()
def pt_evaluation(model_name_or_path, pt_dataset, output_dir, relevance_cutoff=100, split='validation'):
    """
    PyTerrier evaluation -- allows us to evaluate over the entire dataset. Then, we use our
    other evaluation scheme to finally evaluate metrics such as AUC and MRR.

    Params:
    - model_name_or_path: the path for the model
    - pt_dataset: the PyTerrier dataset
    - output_dir: where we'll save the experimental results
    - relevance_cutoff: the @k we want to evaluate for

    Metrics used:
    Precision@k: how many of the top-k items are relevant (TP / (TP + FP))
    Recall@k: how many actual relevant results are shown out of all actual relevant results (TP / (TP + FN)
    
    Returns the final scores for these metrics.
    """

    CACHE_DIR = Path(user_cache_dir("open-mds", "ai2")) / "indices"
    index_path =  CACHE_DIR / pt_dataset.path
    if pt_dataset.name is not None:
        index_path = index_path / pt_dataset.name
    index_path.mkdir(parents=True, exist_ok=True)

    import pyterrier as pt
    from pyterrier_sentence_transformers import SentenceTransformersIndexer, SentenceTransformersRetriever
    indexer = SentenceTransformersIndexer(
        model_name_or_path=model_name_or_path,
        index_path=str(index_path),
        overwrite=False,
        normalize=False,
        verbose=False,
    )
    indexer.index(pt_dataset.get_corpus_iter(verbose=True))
    retrieval_pipeline = SentenceTransformersRetriever(
        model_name_or_path=model_name_or_path,
        index_path=str(index_path),
        num_results=relevance_cutoff,
        verbose=False,
    )
    topics = pt_dataset.get_topics(split)
    qrels = pt_dataset.get_qrels(split)
    retrieved = retrieval_pipeline.transform(topics)
    
    # Metrics: P@1, P@5, P@{relevance_cutoff}, recall@1, recall@5, recall@{relevance_cutoff}
    eval_metrics = ["P_1", "recall_1", "P_5", "recall_5"]
    eval_metrics += [f"P_{relevance_cutoff}", f"recall_{relevance_cutoff}"]

    print(f"[bold]:test_tube: Evaluating retrieved results on the {split} set [/bold]")

    if model_name_or_path == "facebook/contriever":
        name = "Contriever"
    elif model_name_or_path == "facebook/contriever-msmarco":
        name = "Contriever-MSMARCO"
    else:
        name = "Contriever-MS2"

    outputs = pt.Experiment(
        [retrieved],
        topics=topics,
        qrels=qrels,
        eval_metrics=eval_metrics,
        names=[name],
        save_dir=output_dir,
        save_mode="overwrite",
        round=4,
        verbose=True,
    )

    return outputs

            
def main():
    """Trains the retriever on the current dataset"""

    parser = HfArgumentParser(
        (RetrieverFinetuningArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    # Our unique parsing strategy (which depends on OmegaConf) exists here
    elif any(argv.endswith(".yml") for argv in sys.argv[1:]):
        conf = util.parse_omega_conf()
        retriever_args, training_args = parser.parse_dict(conf)
    else:
        retriever_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f" Initializing wandb!")
    wandb.init(
        # set the wandb project where this run will be logged
        project="retriever-finetuning-ms2",

        # track hyperparameters and run metadata
        config={
            "architecture": "Contriever",
            "dataset": "ms2",
            "learning_rate": training_args.learning_rate,
            "train_batch_size": training_args.per_device_train_batch_size,
            "eval_batch_size": training_args.per_device_eval_batch_size,
            "weight_decay": training_args.weight_decay,
            "num_epochs": training_args.num_train_epochs,
            "log_freq": retriever_args.log_freq,
            "relevance_cutoff": retriever_args.relevance_cutoff
        }
    )

    # Setting seed
    set_seed(training_args.seed)

    # Loading dataset, model, tokenizer individually per process
    if retriever_args.hf_dataset_name == Dataset.ms2:
        pt_dataset = indexing_basic.MSLR2022Dataset(name=retriever_args.hf_dataset_name)
    else:
        raise NotImplementedError

    print(
        f"[bold]:book: Dataset chosen: '{retriever_args.hf_dataset_name}'... [/bold]"
    )
    print(
        f"[bold]:computer: Number of GPUs: '{torch.cuda.device_count()}'... [/bold]"
    )
    print(
        f"[bold]:car: Model chosen: {retriever_args.model_name_or_path}... [/bold]", 
    )
    
    model = AutoModel.from_pretrained(retriever_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(retriever_args.model_name_or_path)

    # Put model on DP
    model.cuda()
    model = DataParallel(model)

    # Training here implies also validation
    if training_args.do_train:
        logger.info(f" Starting training!")

        finetune(
            model,
            tokenizer,
            training_args.output_dir,
            pt_dataset,
            retriever_args.n_random_negatives,
            lr=training_args.learning_rate,
            train_batch_size=training_args.per_device_train_batch_size,
            eval_batch_size=training_args.per_device_eval_batch_size,
            weight_decay=training_args.weight_decay,
            num_epochs=training_args.num_train_epochs,
            log_freq=retriever_args.log_freq,
            relevance_cutoff=retriever_args.relevance_cutoff
        )

    # Here, we take the test set and then do the predictions on the eval
    # For do_eval, we include the ROC Curve and final MRR calculations
    if training_args.do_eval:
        logger.info(f" Starting evaluation!")

        pt_dataset_do_eval = copy.deepcopy(pt_dataset)
        del pt_dataset_do_eval._hf_dataset['train']
        del pt_dataset_do_eval._hf_dataset['validation']

        print(
            pt_evaluation(
                retriever_args.model_name_or_path,
                pt_dataset_do_eval,
                training_args.output_dir,
                retriever_args.relevance_cutoff,
                split="test"
            )
        )

        dataset_dict = transform_data(
            pt_dataset,
            n_random_negatives=retriever_args.n_random_negatives,
            splits=['test']
        )
        eval_dataset = dataset_dict['test']

        auc, mrr = evaluate(model, tokenizer, eval_dataset)
        mrr = sum(mrr) / len(mrr)
    
        # Plot this final evaluation
        plt.title("AUC Curve over the test dataset")
        plt.ylabel("True Positives Rate")
        plt.xlabel("False Positive Rate")
        for datapoint in auc:
            fpr, tpr, _ = datapoint
            plt.plot(fpr, tpr, alpha=0.7)
        wandb.log({'AUC Curve over test': plt})

        logger.info(f" Average MRR over the test dataset: {mrr}")

if __name__ == "__main__":
    main()