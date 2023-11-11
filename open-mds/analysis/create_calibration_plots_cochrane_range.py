from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random

from pathlib import Path
from sklearn.linear_model import SGDClassifier

from open_mds import indexing_basic
from open_mds.common import util

from pyterrier_sentence_transformers import SentenceTransformersIndexer, SentenceTransformersRetriever

_DOCUMENT_INDEX_DIR = Path(util.CACHE_DIR) / "indices"

random.seed(42)
np.random.seed(42)

def construct_data(review_ids, pmids, retrieved):
    all_pmid_results = []
    all_neg_pmid_results = []

    new_pmids = []
    new_scores = []
    new_labels = []
    # Construct the data
    for idx, (review, pmid) in enumerate(zip(review_ids, pmids)):
        curr_pmids = []
        # Review ID is key
        docs = retrieved[retrieved.qid == review]
        
        # Match to pmid
        for doc in list(docs.docno):
            if doc in pmid:
                new_pmids.append(doc)
                curr_pmids.append(doc)
                new_scores.append(docs[docs.docno == doc]['score'].item())
                new_labels.append(1)

        not_docs = docs[~docs.docno.isin(curr_pmids)]

        if len(curr_pmids) > len(not_docs):
            neg_sampled = not_docs.sample(n=len(not_docs), random_state=42)
        else:
            neg_sampled = not_docs.sample(n=len(curr_pmids), random_state=42)

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

    logreg = SGDClassifier(
        loss="log_loss",
        penalty=None,
        shuffle=True
    )
    logreg.fit(X, y)

    return logreg

pt_dataset = indexing_basic.MSLR2022Dataset(name="cochrane")

index_path = _DOCUMENT_INDEX_DIR / pt_dataset.path
if pt_dataset.name is not None:
    index_path = index_path / pt_dataset.name
index_path.mkdir(parents=True, exist_ok=True)

indexer = SentenceTransformersIndexer(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    overwrite=False,
    normalize=False,
    verbose=False,
)

indexer.index(pt_dataset.get_corpus_iter(verbose=True))

# Retrieve and train logreg - top-100
retrieval_pipeline = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=100,
    verbose=False,
)

hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
train_topics = pt_dataset.get_topics('train')
train_retrieved = retrieval_pipeline.transform(train_topics)

logreg = train_logreg(hf_dataset, train_retrieved)

pickle.dump(logreg, open("/scratch/li.mil/open-domain-mds-merge/platt_calibration/cochrane/logreg_100.pkl", "wb"))

# Retrieve and train logreg - top-200
retrieval_pipeline_200 = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=200,
    verbose=False,
)

hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
train_topics = pt_dataset.get_topics('train')
train_retrieved = retrieval_pipeline_200.transform(train_topics)

logreg_200 = train_logreg(hf_dataset, train_retrieved)

pickle.dump(logreg_200, open("/scratch/li.mil/open-domain-mds-merge/platt_calibration/cochrane/logreg_200.pkl", "wb"))


# Retrieve and train logreg - top-500
retrieval_pipeline_500 = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=500,
    verbose=False,
)

hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
train_topics = pt_dataset.get_topics('train')
train_retrieved = retrieval_pipeline_500.transform(train_topics)

logreg_500 = train_logreg(hf_dataset, train_retrieved)

pickle.dump(logreg_500, open("/scratch/li.mil/open-domain-mds-merge/platt_calibration/cochrane/logreg_500.pkl", "wb"))

# Retrieve and train logreg - top-1000
retrieval_pipeline_1000 = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=1000,
    verbose=False,
)

hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
train_topics = pt_dataset.get_topics('train')
train_retrieved = retrieval_pipeline_1000.transform(train_topics)

logreg_1000 = train_logreg(hf_dataset, train_retrieved)

pickle.dump(logreg_1000, open("/scratch/li.mil/open-domain-mds-merge/platt_calibration/cochrane/logreg_1000.pkl", "wb"))

# Get the validation data
# 100
retrieval_pipeline_valid = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=100,
    verbose=False,
)

review_ids = hf_dataset['validation']['review_id']
pmids = hf_dataset['validation']['pmid']

valid_topics = pt_dataset.get_topics('validation')
valid_retrieved = retrieval_pipeline_valid.transform(valid_topics)
df = construct_data(review_ids, pmids, valid_retrieved)

# Shuffling
df = df.sample(frac = 1)

features = ["score"]
X = df[features]
y = df.label

# 200
retrieval_pipeline_valid_200 = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=200,
    verbose=False,
)

review_ids = hf_dataset['validation']['review_id']
pmids = hf_dataset['validation']['pmid']

valid_topics_200 = pt_dataset.get_topics('validation')
valid_retrieved_200 = retrieval_pipeline_valid_200.transform(valid_topics_200)
df_200 = construct_data(review_ids, pmids, valid_retrieved_200)

# Shuffling
df_200 = df_200.sample(frac = 1)

features = ["score"]
X_200 = df_200[features]
y_200 = df_200.label

# 500
retrieval_pipeline_valid_500 = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=500,
    verbose=False,
)

review_ids = hf_dataset['validation']['review_id']
pmids = hf_dataset['validation']['pmid']

valid_topics_500 = pt_dataset.get_topics('validation')
valid_retrieved_500 = retrieval_pipeline_valid_500.transform(valid_topics_500)
df_500 = construct_data(review_ids, pmids, valid_retrieved_500)

# Shuffling
df_500 = df_500.sample(frac = 1)

features = ["score"]
X_500 = df_500[features]
y_500 = df_500.label

# 1000
retrieval_pipeline_valid_1000 = SentenceTransformersRetriever(
    model_name_or_path="/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best",
    index_path=str(index_path),
    num_results=1000,
    verbose=False,
)

review_ids = hf_dataset['validation']['review_id']
pmids = hf_dataset['validation']['pmid']

valid_topics_1000 = pt_dataset.get_topics('validation')
valid_retrieved_1000 = retrieval_pipeline_valid_1000.transform(valid_topics_1000)
df_1000 = construct_data(review_ids, pmids, valid_retrieved_1000)

# Shuffling
df_1000 = df_1000.sample(frac = 1)

features = ["score"]
X_1000 = df_1000[features]
y_1000 = df_1000.label

clf_list = [
    (logreg, "Logistic, top-100"),
    (logreg_200, "Logistic, top-200"),
    (logreg_500, "Logistic, top-500"),
    (logreg_1000, "Logistic, top-1000"),
]

data_list = [
    (X, y),
    (X_200, y_200),
    (X_500, y_500),
    (X_1000, y_1000)
]

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")
ax_calibration_curve = fig.add_subplot(gs[:2, :2])

calibration_displays = {}

for i, (clf, name) in enumerate(clf_list):
    # LR display
    display = CalibrationDisplay.from_estimator(
        clf,
        data_list[i][0],
        data_list[i][1],
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i)
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots, for Cochrane dataset (Logistic Regression, top-k)")

grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

fig.tight_layout()
plt.savefig('/home/li.mil/open-domain-mds-merge/open-mds/analysis/calibration_plot_cochrane_range.png', bbox_inches='tight')
