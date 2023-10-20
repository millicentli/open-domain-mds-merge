"""
Extension on open-domain multi-document summarization, except we want to marginalize over
the uncertainty in the dataset.

The pipeline is:
1. Index all of the documents
2. For each query:
--> Come up with a **set** of documents (set k = 2 maybe first, increase slowly, graph performance)
--> Try the strategies (#1 average over docs individually, #2 weighted + average over docs individually)

Q: index the files first or determine them based on relevance?
Example command:
python ./scripts/uncertainty_index_and_retrieve.py "ms2" "./output/datasets/ms2_full" 2


How do we run this?
python ./scripts/uncertainty_index_and_retrieve.py \
        "ms2" "./output/datasets/tiny_ms2_retrieved_split=1" \
        1 \
        --model-name-or-path "/scratch/li.mil/open-domain-mds-merge/contriever_ms2/model_best" 
"""

# 1. Convert all documents into the embeds and store in faiss dataframe
# 2. For each query, build an index of documents associated with the query
# 3. For each query, save that information

import copy
from enum import Enum
from pathlib import Path
from typing import List

from functools import partial
from rich import print
from sklearn.linear_model import LogisticRegression as lr
from transformers import AutoModel, AutoTokenizer

import more_itertools
import numpy as np
import pandas as pd
import random
import torch
import typer

from open_mds import indexing_basic
from open_mds.common import util

app = typer.Typer()

# The default location to save document indices.
_DOCUMENT_INDEX_DIR = Path(util.CACHE_DIR) / "indices"

# The neural retirever to use for dense retireval pipeline. This could be made an argument to the script.
_DEFAULT_NEURAL_RETRIEVER = "facebook/contriever-msmarco"

# Default batch size to use
_BATCH_SIZE = 32

class Dataset(str, Enum):
    multinews = "multinews"
    wcep = "wcep"
    multixscience = "multixscience"
    ms2 = "ms2"
    cochrane = "cochrane"

    """
    For cochrane, target is the query
    For ms2, background is the query
    Check the source of queries for the other datasets
    """

def construct_data(review_ids, pmids, retrieved):
    all_pmid_results = []
    all_neg_pmid_results = []

    new_pmids = []
    new_scores = []
    new_labels = []
    # Construct the data
    for idx, (review, pmid) in enumerate(zip(review_ids, pmids)):
        # Review ID is key
        docs = retrieved[retrieved.qid == review]
        not_docs = retrieved[retrieved.qid != review]
        
        # Match to pmid
        for doc in list(docs.docno):
            if doc in pmid:
                new_pmids.append(doc)
                new_scores.append(docs[docs.docno == doc]['score'].item())
                new_labels.append(1)

        neg_sampled = not_docs.sample(n=len(docs.docno), random_state=42)

        for docno, score in zip(neg_sampled.docno, neg_sampled.score):
            new_pmids.append(docno)
            new_scores.append(score)
            new_labels.append(0)

    # Now, fashion it for training by putting it into a df
    data_dict = {"review_id": new_pmids, "score": new_scores, "label": new_labels}
    df = pd.DataFrame.from_dict(data_dict)

    return df


def train_logreg(hf_dataset, retrieved, split='train'):
    """
    Trained logreg to get the appropriate percentages for the scores

    Takes in the original huggingface dataset and the retrieved dataset;
    fashions the data for training the logreg
    """
    review_ids = hf_dataset[split]['review_id']
    pmids = hf_dataset[split]['pmid']

    assert(len(review_ids) == len(pmids))

    df = construct_data(review_ids, pmids, retrieved)

    # Shuffling
    df = df.sample(frac = 1)

    features = ["score"]
    X = df[features]
    y = df.label

    logreg = lr(random_state=42)
    logreg.fit(X, y)

    return logreg
    

@app.command()
def main(
    hf_dataset_name: Dataset = typer.Argument(
        ..., case_sensitive=False, help="The name of a supported HuggingFace Dataset."
    ),
    output_dir: Path = typer.Argument(
        ...,
        help=("Path to the directory where the dataset and retrieval results will be saved."),
    ),
    index_path: Path = typer.Option(
        None,
        help=(
            "Directory to save the PyTerrier index. If an index already exists at this path and"
            " --overwrite-index is not passed, the index will be overwritten. If not provided, the index will be"
            " saved to util.CACHE_DIR / 'indexes'.")
    ),
    model_name_or_path: str = typer.Option(
        _DEFAULT_NEURAL_RETRIEVER,
        help=(
            "Which model to use for dense retrieval. Can be any Sentence Transformer or HuggingFace Transformer"
            f" model. Defaults to {_DEFAULT_NEURAL_RETRIEVER} Has no effect if choosen retriever does not use a"
            " neural model."
        ),
    ),
    overwrite_cache: bool = typer.Option(
        False, "--overwrite-cache", help="Overwrite the cached copy of the HuggingFace dataset, if it exits."
    ),
    subsets: int = typer.Argument(
        1, # should be ~10 but I'll do 2 for now
        help="Number of subsets per query that we keep"
    ),
    relevance_cutoff: int = typer.Argument(
        100,
        help="Number of documents to cut off for relevance; num_results_per_query basically, so the number of results to retrieve per query. 
             " Only really change if you want to be able to retrieve from a larger swath of documents.
    ),
    # num_docs: int = typer.Argument(
    #     10,
    #     help="Number of docs we want per subsets"
    # ),
    splits: List[str] = typer.Option(
        None, help="Which splits of the dataset to replace with retrieved documents. Defaults to all splits."
    ),
    projection_size: int = typer.Argument(
        768,
        help="Projection size (emb size) for the index",
    ),
    n_subquantizers: int = typer.Argument(
        0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    ),
    n_bits: int = typer.Argument(
        8,
        help="Number of bits per subquantizer",
    ),

) -> None:
    """Recreates the chosen HuggingFace dataset using the documents retrieved from an IR system."""
    
    print(
        f"[bold]:magnifying_glass_tilted_right: Subset set at '{subsets}'... [/bold]"
    )
    print(
        f"[bold]:magnifying_glass_tilted_right: Relevance cutoff set at '{relevance_cutoff}'... [/bold]"
    )

    # Model-specific setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)

    print(f"[bold blue]:information: Model chosen: {model_name_or_path}")

    # Any dataset specific setup goes here
    if hf_dataset_name == Dataset.multinews:
        path = "multi_news"
        doc_sep_token = util.DOC_SEP_TOKENS[path]
        pt_dataset = indexing_basic.CanonicalMDSDataset(path, doc_sep_token=doc_sep_token)
    elif hf_dataset_name == Dataset.wcep:
        path = "ccdv/WCEP-10"
        doc_sep_token = util.DOC_SEP_TOKENS[path]
        pt_dataset = indexing_basic.CanonicalMDSDataset(path, doc_sep_token=doc_sep_token)
    elif hf_dataset_name == Dataset.multixscience:
        pt_dataset = indexing_basic.MultiXScienceDataset()
    elif hf_dataset_name == Dataset.ms2 or hf_dataset_name == Dataset.cochrane:
        pt_dataset = indexing_basic.MSLR2022Dataset(name=hf_dataset_name.value)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create a directory to store the index if it wasn't provided
    index_path = Path(index_path) if index_path is not None else _DOCUMENT_INDEX_DIR / pt_dataset.path
    if pt_dataset.name is not None:
        index_path = index_path / pt_dataset.name
    index_path.mkdir(parents=True, exist_ok=True)
    # Use all splits if not specified
    splits = splits or list(pt_dataset._hf_dataset.keys())
    print(f"[bold blue]:information: Will replace documents in {', '.join(splits)} splits")

    # Create a new copy of the dataset and replace its source documents with retrieved documents
    hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
    print(f"[bold green]:white_check_mark: Loaded the dataset from '{pt_dataset.info_url()}' [/bold green]")

    # Init retriever
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

    docs = pt_dataset.get_corpus_iter(verbose=True)

    print(f"[bold green]:white_check_mark: Loaded the index from '{index_path}' [/bold green]")

    # TODO: need to link query w/ review id
    # What's the eval?
    for split in splits:
        print(
            f"[bold]:magnifying_glass_tilted_right: Retrieving docs for each example in the '{split}' set... [/bold]"
        )
        topics = pt_dataset.get_topics(split)
        retrieved = retrieval_pipeline.transform(topics)

        # Retrieve and train the model
        # We train on train and then extract probs for the other splits
        if split == "train":
            logreg = train_logreg(hf_dataset, retrieved)

        X = pd.DataFrame(retrieved["score"], columns=["score"])
        calibrated = logreg.predict_proba(X)
        retrieved['calibrated'] = calibrated[:, -1]

        # Now, randomly select from list of docs, calculate bern, and see if doc is selected
        def sample_bern(row):
            """
            Bernoulli sampling, returns whether sample is taken.
            """

            rand_likelihood = random.random()
            return 1 if rand_likelihood < row['calibrated'] else 0
        
        # Document selection
        qid_list = set(retrieved.qid)
        qid_to_docnos = {}
        for qid in qid_list:
            selected_docnos = []
            for subset in range(subsets):
                # related = retrieved[retrieved.qid == qid]
                # We want to take a percentage here and sample from most relevant; let's make
                # a cutoff
                # Calculate the percentage
                # num_docs = int((relevance_cutoff / 100) * len(related))
                # related = related.head(num_docs)
                # related = related.sample(frac=1, random_state=42)
                related = retrieved[retrieved.qid == qid].sample(frac=1, random_state=42)
                related['selected'] = related.apply(sample_bern, axis=1)
                selected = related[related.selected == 1]
                selected_docnos.append(list(selected.docno))
            qid_to_docnos[qid] = selected_docnos            
        
        print("After creating dataframe!")
        # Now I have the set of documents; assemble a new dataset that copies the prev
        hf_dataset[split] = hf_dataset[split].map(
            partial(pt_dataset.replace, split=split, qid_to_docnos=qid_to_docnos),
            with_indices=True,
            load_from_cache_file=not overwrite_cache,
            desc=f"Re-building {split} set",
        )
        print(f"[bold blue]:repeat: Source documents in '{split}' set replaced with retrieved documents[/bold blue]")
        
        hf_dataset.save_to_disk(output_dir)
        print(f"[bold green]:floppy_disk: Re-built dataset saved to {output_dir} [/bold green]")

if __name__ == "__main__":
    app()
