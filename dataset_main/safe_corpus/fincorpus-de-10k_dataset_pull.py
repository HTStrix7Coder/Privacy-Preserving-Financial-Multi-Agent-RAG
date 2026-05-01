"""
Load the FinCorpus-DE10k dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/anhaltai/fincorpus-de-10k

FinCorpus-DE10k: ~12k PDFs of financial documents (mostly German), with plaintext for ~10.5k.
Features: filename, text. Configs: Annual_reports, BBK_monthly, Base_prospectuses, Final_terms, Law.
"""

from typing import cast
from datasets import DatasetDict, load_dataset

# Load all collections (default). One instance per .txt document; features: filename, text.
dataset = cast(DatasetDict, load_dataset("anhaltai/fincorpus-de-10k", trust_remote_code=True))

# Inspect structure
print("Dataset:", dataset)
train = dataset["train"]
print("Train size:", len(train))
print("Features:", train.features)
print("\nSample row:")
print(train[0])

# Optional: load a specific collection only
# dataset = load_dataset("anhaltai/fincorpus-de-10k", "Final_terms", trust_remote_code=True)

# Optional: stream for large-scale processing
# streamed = load_dataset("anhaltai/fincorpus-de-10k", split="train", streaming=True, trust_remote_code=True)
# for item in streamed:
#     print(item["filename"], len(item["text"]))
