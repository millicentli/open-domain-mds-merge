"""
Tuned baseline implementation. We take our trained retriever model to get the documents, then we
select and alpha that will give us optimally the best scores possible.

However, we need to get a way to determine which "alpha" is better.
A way to do this evaluation is simply to get the new document splits and do a direct evaluation
via the LM we care about to see if we can maximize its vanilla summarization output. We can
deploy Chantal's strategy of sampling randomly ~500 data points and then validating BertScore
over this to figure out which alpha is the best.

Once the splits are generated, then we use these splits for evaluation on our trained model.
"""

import argparse
import copy
import flatten_dict
import json
import logging
import numpy as np
import pandas as pd
import random
import torch

from dataclasses import dataclass
from datasets import Dataset, load_dataset
from functools import partial
from pathlib import Path
from rich import print
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

from open_mds import indexing_basic
from open_mds import metrics as summarization_metrics
from open_mds.common.util import preprocess_ms2
# TODO: Need to move the functions to a util file
from uncertainty_index_and_retrieve import train_logreg
from open_mds.common import util
from pyterrier_sentence_transformers import SentenceTransformersIndexer, SentenceTransformersRetriever

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configs to use

    dataset_name: the name of the dataset
    dataset_config_name: the specific split of the dataset, which could be "ms2"; otherwise None
    do_train_classifier: whether we need to train our classifier
    do_eval_classifier: whether we need to evaluate our classifier
    output_dir: the output dir for the saved indexed dataset, log

    alpha_range: the range of alphas to test (low, hi) -- hi is max of 0.1
    alpha_step: the step of alphas to try (0.1, for instance)
    """

    dataset_name: str
    dataset_config_name: str
    retriever_name_or_path: str
    model_name_or_path: str
    output_dir: str
    seed: int

    alpha_range: list
    alpha_step: float

def main():
    # Parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = Config(**json.load(f))

    # Set seed
    set_seed(config.seed)

    # Initialize our model of interest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path).to(device)

    print(f"[bold blue]:information: Retriever chosen: {config.retriever_name_or_path}")
    print(f"[bold blue]:information: Model chosen: {config.model_name_or_path}")

    # Load dataset
    pt_dataset = indexing_basic.MSLR2022Dataset(name=config.dataset_config_name)

    # # TODO: remove after debugging
    # pt_dataset._hf_dataset['train'] = Dataset.from_dict(pt_dataset._hf_dataset['train'][:100])
    # pt_dataset._hf_dataset['validation'] = Dataset.from_dict(pt_dataset._hf_dataset['validation'][:10])
    # pt_dataset._hf_dataset['test'] = Dataset.from_dict(pt_dataset._hf_dataset['test'][:10])

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Create a directory to store the index
    _DOCUMENT_INDEX_DIR = Path(util.CACHE_DIR) / "indices"
    index_path = Path(_DOCUMENT_INDEX_DIR / pt_dataset.path)
    if pt_dataset.name is not None:
        index_path = index_path / pt_dataset.name
    index_path.mkdir(parents=True, exist_ok=True)
    # Use all splits if not specified
    splits = list(pt_dataset._hf_dataset.keys())
    print(f"[bold blue]:information: Will replace documents in {', '.join(splits)} splits")

    print(f"[bold green]:white_check_mark: Loaded the dataset from '{pt_dataset.info_url()}' [/bold green]")

    indexer = SentenceTransformersIndexer(
        model_name_or_path=config.retriever_name_or_path,
        index_path=str(index_path),
        overwrite=False,
        normalize=False,
        verbose=False,
    )

    indexer.index(pt_dataset.get_corpus_iter(verbose=True))
    index_total = indexer.faiss_index.index.ntotal
    retrieval_pipeline = SentenceTransformersRetriever(
        model_name_or_path=config.retriever_name_or_path,
        index_path=str(index_path),
        num_results=index_total,
        verbose=False,
    )

    docs = pt_dataset.get_corpus_iter(verbose=True)

    print(f"[bold green]:white_check_mark: Loaded the index from '{index_path}' [/bold green]")

    class Collater(object):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def __call__(self, example):
            input_ids = [ex['input_ids'] for ex in example]
            attention_mask = [ex['attention_mask'] for ex in example]
            labels = [ex['labels'] for ex in example]
            global_attention_mask = [ex['global_attention_mask'] for ex in example]

            return {
                "input_ids": torch.LongTensor(input_ids),
                "attention_mask": torch.LongTensor(attention_mask),
                "labels": torch.LongTensor(labels),
                "global_attention_mask": torch.LongTensor(global_attention_mask)
            }

    def compute_metrics(tokenizer, preds, labels):
        """
        Custom computation of metrics.
        BertScore is our best metric that we compare, but we also use ROUGE.
        """

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_results = summarization_metrics.compute_rouge(predictions=decoded_preds, references=decoded_labels)
        bertscore_results = summarization_metrics.compute_bertscore(
            predictions=decoded_preds,
            references=decoded_labels,
            device=preds[0].device
        )

        results = {
            **flatten_dict.flatten(rouge_results, reducer="underscore"),
            **flatten_dict.flatten({"bertscore": bertscore_results}, reducer="underscore"),
        }

        return results

    def preprocess(tokenizer, examples):
        """
        Preprocesses the documents and appends them into a single example, for validating.
        """

        # Do one at a time
        text_list = []
        summary_list = []
        for background, title, abstract, target in zip(examples['background'], examples['title'], examples['abstract'], examples['target']):
            b = background.strip()
            articles = [f"{ti.strip()} {ab.strip()}" for ti, ab in zip(title[0], abstract[0])]
            text = f" {tokenizer.sep_token} ".join([b] + articles) 
            summary = target.strip()
            text_list.append(text)
            summary_list.append(summary)

        # TODO: don't hardcode max length here
        model_inputs = tokenizer(text_list, max_length=16384, padding=True, truncation=True)

        labels = tokenizer(text_target=summary_list, max_length=256, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        global_attention_tokens = [tokenizer.sep_token]
        if tokenizer.bos_token is not None:
            global_attention_tokens.append(tokenizer.bos_token)
        model_inputs["global_attention_mask"] = util.get_global_attention_mask(
            model_inputs.input_ids,
            token_ids=tokenizer.convert_tokens_to_ids(global_attention_tokens),
        )

        logger.info(f"Using global attention on the following tokens: {global_attention_tokens}")
        return model_inputs


    @torch.no_grad()
    def score(model, tokenizer, dataset):
        """
        Scores the dataset according to what we care about, validating over 500 points (per Chantal's paper)
        """

        indices = np.arange(len(dataset['validation']))
        chosen_indices = random.sample(list(indices), 2)
        subdataset = dataset['validation'].select(chosen_indices)

        collater = Collater(tokenizer)
        sampler = SequentialSampler(subdataset)
        
        col_names = ["review_id", "pmid", "title", "abstract", "target", "background"]

        subdataset = subdataset.map(
            partial(preprocess, tokenizer),
            batched=True,
            num_proc=1,
            remove_columns=col_names,
            desc="Running tokenizer on validation set",
            batch_size=None
        )
        dataloader = DataLoader(
            subdataset,
            sampler=sampler,
            collate_fn=collater
        )

        all_preds = []
        all_refs = []
        for batch in tqdm(dataloader):
            batch = {key: batch[key].to(device) for key in batch}
            predictions = model.generate(
                input_ids=batch['input_ids'],
                num_beams=2,
                max_new_tokens=256,
                no_repeat_ngram_size=3
            )
            all_preds.extend(predictions)
            all_refs.extend(batch['labels'])


        return compute_metrics(tokenizer, all_preds, all_refs)

    train_topics = pt_dataset.get_topics("train")
    train_retrieved = retrieval_pipeline.transform(train_topics)
    logreg = train_logreg(pt_dataset._hf_dataset, train_retrieved)

    # First, we only need to index on our validation set
    split = "validation"
    print(
        f"[bold]:magnifying_glass_tilted_right: Retrieving docs for each example in the '{split}' set... [/bold]"
    )
    topics = pt_dataset.get_topics(split)
    retrieved = retrieval_pipeline.transform(topics)
    X = pd.DataFrame(retrieved["score"], columns=["score"])
    calibrated = logreg.predict_proba(X)
    retrieved['calibrated'] = calibrated[:, -1]
    
    # Invert scores
    retrieved['inverted_calibrated'] = 1 - retrieved['calibrated']

    qid_list = set(retrieved.qid)
    qid_to_docnos = {}
    distr = np.arange(float(config.alpha_range[0]), float(config.alpha_range[1]), float(config.alpha_step))
    curr_best = -10000
    best_alpha = -1
    alpha_scores = []
    for alpha in distr:
        for qid in qid_list:
            related = retrieved[retrieved.qid == qid] 
            related = related[related.inverted_calibrated < alpha]

            # If len(related) is none, then during train time we give this 0% prob
            if len(related) == 0:
                selected_docnos = []
            # Else we have some docs
            else:
                selected_docnos = list(related.docno)
            
            qid_to_docnos[qid] = [selected_docnos]

        # Create the dataset
        hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
        hf_dataset[split] = hf_dataset[split].map(
            partial(pt_dataset.replace, split="validation", qid_to_docnos=qid_to_docnos),
            with_indices=True,
            desc="Re-building validation split with alpha = %0.2f" % alpha
        )

        # Now, run the test; based on Chantal's paper which claims 500 points is enough to assess
        # the performance of a model on a specific dataset
        results = score(model, tokenizer, hf_dataset)
        if results['bertscore_f1_mean'] > curr_best:
            curr_best = results['bertscore_f1_mean']
            best_alpha = alpha

            logger.info(f"Current best alpha: {best_alpha}")

        alpha_scores.append(results['bertscore_f1_mean'])

    # At the end of this, output all alphas into a file to save
    alpha_dict = {
        "%0.2f" % alpha: score for alpha, score in zip(distr, alpha_scores)
    }

    with open(Path(config.output_dir) / "scores.json", "w") as f:
        json.dump(alpha_dict, f)

    # Now, we save a new copy of the best dataset
    hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
    for qid in qid_list:
        related = retrieved[retrieved.qid == qid] 
        related = related[related.inverted_calibrated < best_alpha]

        # If len(related) is none, then during train time we give this 0% prob
        if len(related) == 0:
            selected_docnos = []
        # Else we have some docs
        else:
            selected_docnos = list(related.docno)
        
        qid_to_docnos[qid] = [selected_docnos]

    hf_dataset[split] = hf_dataset[split].map(
        partial(pt_dataset.replace, split="validation", qid_to_docnos=qid_to_docnos),
        with_indices=True,
        desc="Re-building validation split with best alpha = %0.2f" % best_alpha
    )

    hf_dataset.save_to_disk(config.output_dir)
    print(f"[bold green]:floppy_disk: Re-built dataset saved to {config.output_dir} [/bold green]")

if __name__ == "__main__":
    main()