import requests
import pandas as pd
# from utils.foldseek_util import get_struc_seq


def download_alphafold_pdb_v4(uniprot_id, save_path):
    # Construct the URL for AlphaFold v4 PDB file using the UniProt ID
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    # Send a GET request to fetch the PDB file
    response = requests.get(url)

    # Check if the response is successful
    if response.status_code == 200:
        # Save the PDB file to the specified path
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded AlphaFold PDB for {uniprot_id} to {save_path}")
    else:
        print(f"Failed to download AlphaFold PDB for {uniprot_id}. Status code: {response.status_code}")


# ligands_filepath = "LMDB/PDBBind/LP_PDBBind_edited.csv"
# pdbbind_df = pd.read_csv(ligands_filepath, index_col=0)
# pdbbind_df = pdbbind_df.dropna(subset=["smiles", "uniprot_id"])
#
# for uniprot_id in pdbbind_df["uniprot_id"].values.tolist():
#     save_path = f"example/AF-{uniprot_id}-F1-model_v4.pdb"  # Specify the path where you want to save the PDB file
#     download_alphafold_pdb_v4(uniprot_id, save_path)

# def download_pdb(pdb_id):
#     url = f"https://files.rcsb.org/download/{pdb_id}.cif"
#     response = requests.get(url)
#
#     if response.status_code == 200:
#         pdb_path = f"example/test-{pdb_id}.cif"
#         with open(pdb_path, 'wb') as f:
#             f.write(response.content)
#         return pdb_path
#     else:
#         raise ValueError(f"Error downloading PDB file for ID {pdb_id}: {response.status_code}")
#
# download_pdb("8ac8")

# pdb_path = "example/AF-P30405-F1-model_v4.pdb"  # ""example/8ac8.cif"
#
# # Extract the "A" chain from the pdb file and encode it into a struc_seq
# # pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
# # use AF2 structures for best performance.
# parsed_seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"], plddt_mask=True)["A"]
# seq, foldseek_seq, combined_seq = parsed_seqs
#
# print(f"seq: {seq}")
# print(f"foldseek_seq: {foldseek_seq}")
# print(f"combined_seq: {combined_seq}")

import os
import lmdb
import json
import random
import pandas as pd
from utils.foldseek_util import get_struc_seq

# Load dataset
ligands_filepath = "LMDB/PDBBind/LP_PDBBind_edited.csv"
df = pd.read_csv(ligands_filepath, index_col=0)
df = df.dropna(subset=["smiles", "uniprot_id", "value"])  # Ensure only necessary columns

# Get valid proteins from PDB files
valid_uniprot_ids = {filename.split("-")[1] for filename in os.listdir("example") if ".pdb" in filename}
df = df[df["uniprot_id"].isin(valid_uniprot_ids)]

# Split dataset by unique protein IDs
unique_proteins = list(df["uniprot_id"].unique())
print("nubmer of unique proteins:", len(unique_proteins))
random.shuffle(unique_proteins)

split_idx = int(len(unique_proteins) * 0.95)
train_ids = set(unique_proteins[:split_idx])
val_ids = set(unique_proteins[split_idx:])

train_df = df[df["uniprot_id"].isin(train_ids)]
val_df = df[df["uniprot_id"].isin(val_ids)]

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
print(f"Total size: {len(train_df) + len(val_df)}")
# Function to save data to LMDB


def save_to_lmdb(df, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=int(1e9))
    ii = 0
    with env.begin(write=True) as txn:
        for i, row in df.iterrows():
            uniprot_id = row["uniprot_id"]
            ligand = row["smiles"]
            ba_value = float(row["value"])  # Only using the "value" column

            save_path = f"example/AF-{uniprot_id}-F1-model_v4.pdb"
            parsed_seqs = get_struc_seq("bin/foldseek", save_path, ["A"], plddt_mask=True)["A"]
            seq, foldseek_seq, combined_seq = parsed_seqs  # Extract three types of sequences

            entry = {
                "uniprot_id": uniprot_id,
                "sequence": combined_seq,
                "original_seq": seq,
                "foldseek_seq": foldseek_seq,
                "ligand": ligand,
                "binding_affinity": ba_value
            }

            new_key = f"{ii}".encode("utf-8")

            txn.put(new_key, json.dumps(entry).encode())
            ii += 1
            print("current index:", ii)

        txn.put(b"length", str(len(df)).encode())
    env.close()
    print(f"Saved {len(df)} entries to {lmdb_path}")


# save_to_lmdb(val_df, "pdbbind_val.lmdb")
# save_to_lmdb(train_df, "pdbbind_train.lmdb")


import lmdb
import pickle

# Function to read from LMDB and print

def replace_keys_with_indices(lmdb_path, output_lmdb_path):
    # Open the input LMDB database for reading
    env = lmdb.open(lmdb_path, readonly=True)

    # Create a new LMDB environment for output
    output_env = lmdb.open(output_lmdb_path, map_size=1099511627776)  # Adjust map_size if necessary

    with env.begin() as txn:
        # Create a transaction to read from the input database
        cursor = txn.cursor()

        with output_env.begin(write=True) as out_txn:
            # Iterate over the LMDB records
            ii = 0
            for key, value in cursor:
                # Load the data (assuming it's JSON serializable)
                # print("key", key)
                # print("value", value)

                data = json.loads(value.decode('utf-8'))

                # Create a new key based on index
                new_key = f"{ii}".encode("utf-8")

                # Replace the old key with the new one and serialize the data back to JSON
                out_txn.put(new_key, json.dumps(data).encode('utf-8'))  # Encoding back to bytes

                ii += 1

            out_txn.put(b"length", str(ii).encode())

    # Close both the input and output LMDB environments
    env.close()
    output_env.close()

    print(f"Processed data saved to {output_lmdb_path}")


# Example usage
lmdb_path = 'pdbbind_val.lmdb'
output_lmdb_path = 'new_pdbbind_val.lmdb'
# replace_keys_with_indices(lmdb_path, output_lmdb_path)

lmdb_path = 'pdbbind_train.lmdb'
output_lmdb_path = 'new_pdbbind_train.lmdb'
# replace_keys_with_indices(lmdb_path, output_lmdb_path)


def read_from_lmdb(filename):
    ii = 0
    with lmdb.open(filename) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                # Assuming value is a JSON object, not pickled
                data = json.loads(value.decode('utf-8'))

                if isinstance(data, int) or isinstance(value, int):
                    print(value)

                ii += 1
    print("Data size:", ii)


def add_length_to_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=False, map_size=int(5e9))  # Open in write mode

    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        total_entries = sum(1 for _ in cursor)  # Count total entries

        txn.put(b"length", str(total_entries).encode())  # Store length as a key
        print(f"Added length: {total_entries}")

    env.close()
    print(f"Updated LMDB: {lmdb_path}")


# add_length_to_lmdb('LMDB/PDBBind/valid')


import lmdb
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_human_ppi(lmdb_paths):
    """Loads and merges data from multiple LMDB files."""
    merged_data = []

    for lmdb_path in lmdb_paths:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, map_size=int(5e9))
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key == b"length" or key == b"info":  # Skip length entry
                    continue
                # print("key:", key)
                entry = json.loads(value.decode('utf-8'))
                # print("entry:", entry)
                merged_data.append(entry)
        env.close()

    print(f"Loaded {len(merged_data)} total.")
    return merged_data


def save_lmdb(data, output_lmdb_path):
    """Saves the dataset to an LMDB file."""
    env = lmdb.open(output_lmdb_path, map_size=int(5e9))
    with env.begin(write=True) as txn:
        for i, entry in enumerate(data):
            txn.put(str(i).encode(), json.dumps(entry).encode())
        txn.put(b"length", str(len(data)).encode())  # Store dataset length
    env.close()
    print(f"Saved {len(data)} entries to {output_lmdb_path}")


def filter_and_resplit_human_ppi(lmdb_paths, ligand_df, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Merges, filters, and splits the HumanPPI dataset based on ligand protein IDs.

    :param lmdb_paths: List of LMDB file paths (train, valid, test)
    :param ligand_df: DataFrame with valid UniProt IDs
    :param output_dir: Directory to save new LMDB files
    :param train_ratio: Train split ratio
    :param val_ratio: Validation split ratio
    :param test_ratio: Test split ratio
    """
    # Load and merge data
    data = load_human_ppi(lmdb_paths)

    # Get valid proteins from ligand dataset
    valid_proteins = set(ligand_df["uniprot_id"].unique())

    filtered_data = [
        entry for entry in data if entry["name"] in valid_proteins
    ]

    print(f"Filtered dataset size: {len(filtered_data)}.")

    # Shuffle data
    np.random.shuffle(filtered_data)

    # Split into train, valid, test
    train_data, temp_data = train_test_split(filtered_data, test_size=(val_ratio + test_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    print(f"New Train: {len(train_data)}, Valid: {len(val_data)}, Test: {len(test_data)}")

    # Ensure the directories exist before saving LMDB files
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/valid", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    # Save new splits as LMDB
    save_lmdb(train_data, f"{output_dir}/train")
    save_lmdb(val_data, f"{output_dir}/valid")
    save_lmdb(test_data, f"{output_dir}/test")


# Example Usage
ligand_df = df
lmdb_paths = [
    "LMDB/DeepLoc/cls10/foldseek/train",
    "LMDB/DeepLoc/cls10/foldseek/valid",
    "LMDB/DeepLoc/cls10/foldseek/test"
]
filter_and_resplit_human_ppi(lmdb_paths, ligand_df, "LMDB/DeepLoc/cls10/ligands")

# val_lmdb = "pdbbind_val.lmdb"
# read_from_lmdb("pdbbind_train.lmdb")
