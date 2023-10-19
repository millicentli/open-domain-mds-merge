
"""
Revamped, not PyTerrier anymore
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import pyterrier as pt
from datasets import load_dataset, DatasetDict, Dataset
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from transformers.utils import is_offline_mode

from open_mds.common import util

_HF_DATASETS_URL = "https://huggingface.co/datasets"

if not pt.started():
    # This is a bit of a hack, but the version and helper version are required if you want to use PyTerrier
    # offline. See: https://pyterrier.readthedocs.io/en/latest/installation.html#pyterrier.init
    if is_offline_mode():
        version, helper_version = util.get_pyterrier_versions()
        pt.init(version=version, helper_version=helper_version)
    else:
        pt.init()

class HuggingFacePyTerrierDataset(pt.datasets.Dataset):
    """Simple wrapper for the PyTerrier Dataset class to make it easier to interface with HuggingFace Datasets."""

    def __init__(self, path: str, name: Optional[str] = None, **kwargs) -> None:
        self.path = path
        self.name = name
        # breakpoint()
        self._hf_dataset = load_dataset(self.path, self.name, **kwargs)
        # datasets = load_dataset(self.path, self.name, split=['train[:25%]', 'validation[:25%]', 'test[:25%]'], **kwargs)
        # datasets = load_dataset(self.path, self.name, split=['train[:01%]', 'validation[:01%]', 'test[:01%]'], **kwargs)
        # datasets = load_dataset(self.path, self.name, split=['train[:02%]', 'validation[:02%]', 'test[:02%]'], **kwargs)

        # from datasets import DatasetDict
        # self._hf_dataset = DatasetDict()
        # self._hf_dataset['train'] = datasets[0]
        # self._hf_dataset['validation'] = datasets[1]
        # self._hf_dataset['test'] = datasets[2]

    def replace(
        self, example: Dict[str, Any], idx: int, *, split: str, retrieved: pd.DataFrame, k: Optional[int] = None
    ) -> Dict[str, Any]:
        """This method replaces the original source documents of an `example` from a HuggingFace dataset with the
        top-`k` documents in `retrieved`. It is expected that this function will be passed to the `map` method of
        the HuggingFace Datasets library with the argument `with_indices=True`. If `k` is `None`, it will be set
        dynamically for each example as the original number of source documents. Must be implemented by child class.
        """
        raise NotImplementedError("Method 'replace' must be implemented by the child class.")

    def get_corpus_iter(self, verbose: bool = True) -> Iterator[Dict[str, Any]]:
        """Returns an iterator that yields dictionaries with the keys "docno" and "text" for each example in the
        dataset. Must be implemented by child class.
        """
        raise NotImplementedError("Method 'get_corpus_iter' must be implemented by the child class.")

    def get_topics(self, split: str, max_examples: Optional[int] = None) -> pd.DataFrame:
        """Returns a Pandas DataFrame with the topics (queries) for the given `split`. If `max_examples` is provided,
        only this many topics will be returned. Must be implemented by child class."""
        raise NotImplementedError("Method 'get_topics' must be implemented by the child class.")

    def get_qrels(self, split: str) -> pd.DataFrame:
        """Returns a Pandas DataFrame with the qrels for the given `split`. Must be implemented by child class."""
        raise NotImplementedError("Method 'get_qrels' must be implemented by the child class.")

    def get_index(self, index_path: str, overwrite: bool = False, verbose: bool = True, **kwargs) -> pt.IndexRef:
        """Returns the `IndexRef` for this dataset from `index_path`, creating it first if it doesn't already
        exist. If `overwrite`, the index will be rebuilt. Any provided **kwargs are passed to `pt.IterDictIndexer`.
        """
        if any(Path(index_path).iterdir()):
            if overwrite:
                shutil.rmtree(index_path)
            else:
                return pt.IndexRef.of(index_path)

        indexer = _get_iter_dict_indexer(index_path, dataset=self, **kwargs)

        # Compose index from iterator
        # See: https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html#iterdictindexer
        indexref = indexer.index(self.get_corpus_iter(verbose=verbose))
        return indexref

    def get_document_stats(self, avg_tokens_per_doc=False, avg_tokens_per_summary=False, **kwargs) -> Dict[str, float]:
        """Returns a dictionary with corpus statistics for the given dataset. Must be implemented by child class.
        If avg_tokens_per_doc is True, the average number of tokens per document will be returned.
        """
        raise NotImplementedError("Method 'get_document_stats' must be implemented by the child class.")

    def info_url(self) -> str:
        return f"{_HF_DATASETS_URL}/{self.path}"


class MSLR2022Dataset(HuggingFacePyTerrierDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__("allenai/mslr2022", **kwargs)

        # Collect all documents in the dataset in a way thats easy to lookup
        self._documents = {}
        for split in self._hf_dataset:
            for example in self._hf_dataset[split]:
                for docno, title, abstract in zip(example["pmid"], example["title"], example["abstract"]):
                    self._documents[docno] = {"title": title, "abstract": abstract}


    def replace(
        self, example: Dict[str, Any], idx: int, *, split: str, qid_to_docnos: Dict[str, int]
    ) -> Dict[str, Any]:
        qid = example["review_id"]
        docnos = qid_to_docnos[qid]
        example["pmid"] = docnos
        example["title"] = [[self._documents[docno]["title"] for docno in docno_list] for docno_list in docnos]
        example["abstract"] = [[self._documents[docno]["abstract"] for docno in docno_list] for docno_list in docnos]
        return example

    def get_corpus_iter(self, verbose: bool = False):
        yielded = set()
        for split in self._hf_dataset:
            for example in tqdm(
                self._hf_dataset[split],
                desc=f"Indexing {split}",
                total=len(self._hf_dataset[split]),
                disable=not verbose,
            ):
                for title, abstract, pmid in zip(example["title"], example["abstract"], example["pmid"]):
                    title = title.strip()
                    abstract = abstract.strip()
                    # Don't index duplicate or empty documents
                    if pmid in yielded or not title + abstract:
                        continue
                    yielded.add(pmid)
                    yield {"docno": pmid, "text": f"{title} {abstract}"}

    def get_topics(self, split: str, max_examples: Optional[int] = None) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        if max_examples:
            dataset = dataset[:max_examples]
        # Cochrane does not contain a background section, so use the target as query instead
        queries = dataset["background"] if self.name == "ms2" else dataset["target"]
        qids = dataset["review_id"]
        topics = pd.DataFrame({"qid": qids, "query": queries})
        # return _sanitize_query(topics)
        return topics

    def get_qrels(self, split: str) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        qids, docnos = [], []
        for example in dataset:
            qids.extend([example["review_id"]] * len(example["pmid"]))
            docnos.extend(example["pmid"])
        labels = [1] * len(qids)
        return pd.DataFrame({"qid": qids, "docno": docnos, "label": labels})