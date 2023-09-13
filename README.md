# Initial experiments on merging multiple documents under uncertainty
To begin, follow the orginal setup from the work on [Open Domain Multi-document Summarization](https://github.com/allenai/open-mds#installation). We base this current repo off of their work.

The main difference between ours and theirs is that we introduce a retriever pipeline that retrieves documents under uncertainty up to some k, where k is the number of subsets of documents per query. To run our code:

1. First, create the document index. This is similar to the other work, but run with a different command. The example command is below.
> python ./scripts/uncertainty_index_and_retrieve.py "ms2" "./output/datasets/ms2_split=2" 2

The three args following the Python file is the dataset used, the output directory, and the number of splits determined. The document splits should be saved locally for later use.

2. Next, summarize over the document index created. The command to run is below:
> python ./scripts/uncertainty_run_summarization.py "./conf/base.yml" "./conf/ms2/led-base/eval.yml" \
> &nbsp; &nbsp; &nbsp; &nbsp; output_dir="./output/ms2_split=2/led-base/" \
> &nbsp; &nbsp; &nbsp; &nbsp; dataset_name="./output/datasets/ms2_split=2/"

The first two args are the configs used in the original setup. The last two (`output_dir`, `dataset_name`) are fairly self-explanatory. dataset_name refers to the dataset we created in part 1.

 
