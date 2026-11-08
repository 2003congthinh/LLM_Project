import os
import lzma
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import random

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".parquet") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "openwebtext/plain_text"
output_file_train = "train_split.txt"
output_file_val = "val_split.txt"
vocab_file = "vocab.txt"

# IMPORTANT: Parquet files are tabular. You must specify which column holds the text data.
# Most OpenWebText datasets use 'text' as the column name. Change this if yours is different.
text_column = "text"

files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate split index
split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

# Process the files for training and validation separately
vocab = set()

# Process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        # 1. Read the parquet file into a pandas DataFrame
        df = pd.read_parquet(file_path)
        # 2. Extract the text column, drop empty rows, and join into a single string
        text = "\n".join(df[text_column].dropna().astype(str).tolist())
        outfile.write(text + "\n")
        # 3. Update vocabulary
        vocab.update(set(text))

        # with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        #     text = infile.read()
        #     outfile.write(text)
        #     characters = set(text)
        #     vocab.update(characters)

# Process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_parquet(file_path)
        text = "\n".join(df[text_column].dropna().astype(str).tolist())
        outfile.write(text + "\n")
        vocab.update(set(text))
        # with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        #     text = infile.read()
        #     outfile.write(text)
        #     characters = set(text)
        #     vocab.update(characters)

# Write the vocabulary to vocab.txt
with open(vocab_file, "w", encoding="utf_8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')