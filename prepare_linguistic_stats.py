"""Skrip opsional: menghitung ulang linguistic_mean / linguistic_std untuk model kode ke-3.

Model kode ke-3 (banding-new-skripsi) memakai 30 fitur linguistik yang
distandardisasi memakai mean/std dari TRAIN set. File hasil training di zip
tidak menyertakan mean/std ini, jadi skrip ini menghitungnya dari dataset asli
dengan mengikuti pipeline notebook:

  1. load dataset (xlsx/csv/zip berisi xlsx) kolom text & label
  2. normalisasi label + minimal_clean_text
  3. exact duplicate removal
  4. near-duplicate grouping (datasketch MinHash)
  5. group-aware train/val/test split (StratifiedGroupKFold)
  6. ekstraksi fitur linguistik (spaCy + distilgpt2)
  7. simpan linguistic_mean.npy & linguistic_std.npy ke folder model

Catatan:
- Untuk dataset 33k baris, ekstraksi fitur penuh membutuhkan waktu sangat lama
  di CPU (kira-kira 18 jam, didominasi perplexity distilgpt2). Gunakan
  --sample-size (mis. 1000) untuk estimasi cepat (±30 menit) yang hasilnya
  sudah sangat dekat dengan statistik penuh.
- Website otomatis memakai file linguistic_mean.npy/std.npy bila tersedia.
  Tanpa file tersebut, model kode ke-3 berjalan dengan fitur linguistik netral
  (0 = rata-rata fitur terstandarisasi).

Contoh:
    python prepare_linguistic_stats.py --dataset archive.zip --model-dir models/kode_ke-3 --sample-size 1000
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from model_inference import (
    _ensure_linguistic_components,
    extract_linguistic_features,
    load_model_bundle,
    minimal_clean_text,
)

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE_FROM_TOTAL = 0.15


# ---------------------------------------------------------------------------
# Fungsi preprocessing yang diambil dari notebook (skripsi-1 / banding)
# ---------------------------------------------------------------------------
def normalize_label(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric_value = int(value)
        return numeric_value if numeric_value in (0, 1) else np.nan

    value = str(value).strip().lower()
    human_labels = {
        "0", "human", "human-written", "human written",
        "real", "person", "student",
    }
    ai_labels = {
        "1", "ai", "ai-generated", "ai generated",
        "machine", "llm", "chatgpt",
    }
    if value in human_labels:
        return 0
    if value in ai_labels:
        return 1
    return np.nan


def normalized_text_for_hash(text):
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def make_text_hash(text):
    normalized = normalized_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_ngrams(text, ngram_size=3):
    words = normalized_text_for_hash(text).split()
    if len(words) <= ngram_size:
        return {" ".join(words)}
    return {
        " ".join(words[index : index + ngram_size])
        for index in range(len(words) - ngram_size + 1)
    }


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second
        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first
        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def create_near_duplicate_groups(
    texts, threshold=0.90, num_perm=64, word_ngram_size=3
):
    from datasketch import MinHash, MinHashLSH
    from tqdm.auto import tqdm

    union_find = UnionFind(len(texts))
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for index, text in enumerate(
        tqdm(texts, desc="Near-duplicate grouping")
    ):
        shingles = word_ngrams(text, ngram_size=word_ngram_size)
        minhash = MinHash(num_perm=num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))

        candidates = lsh.query(minhash)
        for candidate in candidates:
            union_find.union(index, int(candidate))
        lsh.insert(str(index), minhash)

    roots = [union_find.find(index) for index in range(len(texts))]
    root_to_group = {}
    group_ids = []
    for root in roots:
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_ids.append(root_to_group[root])
    return np.asarray(group_ids)


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xlsx", ".xls", ".csv"))
            ]
            if not candidates:
                raise ValueError("Zip tidak berisi file dataset.")
            with archive.open(candidates[0]) as file:
                return read_dataset_file(file, Path(candidates[0]).suffix)
    return read_dataset_file(path, path.suffix)


def read_dataset_file(file, suffix: str) -> pd.DataFrame:
    if suffix.lower() == ".csv":
        data = pd.read_csv(file, low_memory=False)
    elif suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(file)
    else:
        raise ValueError(f"Format tidak didukung: {suffix}")

    data.columns = [str(column).strip().lower() for column in data.columns]
    required = {"text", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")
    return data[["text", "label"]].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path dataset (xlsx/xls/csv/zip berisi file tersebut).",
    )
    parser.add_argument(
        "--model-dir",
        default="models/kode_ke-3",
        help="Folder model kode ke-3 tempat mean/std disimpan.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Jumlah sampel train yang dipakai untuk estimasi mean/std. "
        "Gunakan 0 untuk seluruh train set (sangat lambat di CPU).",
    )
    parser.add_argument(
        "--skip-near-duplicate-grouping",
        action="store_true",
        help="Lewati MinHash near-duplicate grouping (jauh lebih cepat). "
        "Hasil estimasi mean/std hampir sama karena statistik fitur "
        "tidak sensitif terhadap perbedaan kecil keanggotaan train set.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print("Memuat dataset:", args.dataset)
    df = load_dataset(args.dataset)
    print("Ukuran awal:", df.shape)

    # Pipeline notebook
    df["label"] = df["label"].apply(normalize_label)
    df["text_raw"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text_raw"].apply(minimal_clean_text)
    before_invalid = len(df)
    df = df[
        df["label"].isin([0, 1]) & df["text_clean"].str.len().gt(0)
    ].copy()
    df["label"] = df["label"].astype(int)
    print("Baris invalid/kosong dibuang:", before_invalid - len(df))

    df["text_hash"] = df["text_clean"].apply(make_text_hash)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text_hash"], keep="first").reset_index(
        drop=True
    )
    print("Exact duplicate dibuang:", before_dedup - len(df))

    if args.skip_near_duplicate_grouping:
        print("Near-duplicate grouping dilewati (mode cepat).")
        df["group_id"] = np.arange(len(df))
    else:
        print("Near-duplicate grouping (threshold 0.90, num_perm 64)...")
        df["group_id"] = create_near_duplicate_groups(
            df["text_clean"].tolist(),
            threshold=0.90,
            num_perm=64,
            word_ngram_size=3,
        )

        conflicting_groups = set(
            df.groupby("group_id")["label"].nunique()
            .loc[lambda s: s > 1]
            .index
        )
        print("Conflicting groups:", len(conflicting_groups))
        if conflicting_groups:
            before_conflict = len(df)
            df = df[~df["group_id"].isin(conflicting_groups)].copy()
            print("Baris conflicting dibuang:", before_conflict - len(df))
        df = df.reset_index(drop=True)

    if args.skip_near_duplicate_grouping:
        # Split stratifikasi biasa (group_id unik per baris).
        from sklearn.model_selection import train_test_split

        train_val, test = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df["label"],
            random_state=SEED,
        )
        relative_validation_size = VAL_SIZE_FROM_TOTAL / (1.0 - TEST_SIZE)
        train, validation = train_test_split(
            train_val,
            test_size=relative_validation_size,
            stratify=train_val["label"],
            random_state=SEED,
        )
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)
    else:
        # Group-aware train/val/test split (sama dengan notebook banding)
        from sklearn.model_selection import StratifiedGroupKFold

        first_splitter = StratifiedGroupKFold(
            n_splits=7, shuffle=True, random_state=SEED
        )
        train_val_indices, test_indices = next(
            first_splitter.split(df, y=df["label"], groups=df["group_id"])
        )
        train_val = df.iloc[train_val_indices].copy()
        test = df.iloc[test_indices].copy()

        second_splitter = StratifiedGroupKFold(
            n_splits=6, shuffle=True, random_state=SEED + 1
        )
        train_indices, validation_indices = next(
            second_splitter.split(
                train_val, y=train_val["label"], groups=train_val["group_id"]
            )
        )
        train = train_val.iloc[train_indices].copy()
        validation = train_val.iloc[validation_indices].copy()

    print(
        f"Split -> train: {len(train)}, validation: {len(validation)}, "
        f"test: {len(test)}"
    )

    # Verifikasi konsistensi dengan npz hasil training model kode ke-3.
    def cache_digest(split_data: pd.DataFrame) -> str:
        digest_source = "|".join(
            split_data["text_hash"].astype(str).tolist()
        )
        return hashlib.md5(digest_source.encode("utf-8")).hexdigest()[:12]

    print("Digest train     :", cache_digest(train))
    print("Digest validation:", cache_digest(validation))
    print("Digest test      :", cache_digest(test))

    sample = train
    if args.sample_size and args.sample_size > 0:
        n_human = min(args.sample_size // 2, int((train["label"] == 0).sum()))
        n_ai = min(args.sample_size - n_human, int((train["label"] == 1).sum()))
        human_sample = train[train["label"] == 0].sample(
            n=n_human, random_state=SEED
        )
        ai_sample = train[train["label"] == 1].sample(
            n=n_ai, random_state=SEED
        )
        sample = pd.concat([human_sample, ai_sample], ignore_index=True)
        print(f"Menggunakan sampel train: {len(sample)} baris")

    print("Memuat komponen linguistik (spaCy + distilgpt2)...")
    bundle = load_model_bundle(model_dir)
    _ensure_linguistic_components(bundle)

    print("Mengekstraksi fitur linguistik...")
    features = []
    for index, (_, row) in enumerate(sample.iterrows(), start=1):
        features.append(
            extract_linguistic_features(
                row["text_raw"],
                nlp=bundle.spacy_nlp,
                perplexity_tokenizer=bundle.perplexity_tokenizer,
                perplexity_model=bundle.perplexity_model,
                enable_perplexity=True,
            )
        )
        if index % 100 == 0:
            print(f"  {index}/{len(sample)} selesai")

    matrix = np.vstack(features).astype(np.float32)
    linguistic_mean = matrix.mean(axis=0)
    linguistic_std = matrix.std(axis=0)
    linguistic_std[linguistic_std < 1e-6] = 1.0

    mean_path = model_dir / "linguistic_mean.npy"
    std_path = model_dir / "linguistic_std.npy"
    np.save(mean_path, linguistic_mean)
    np.save(std_path, linguistic_std)
    print("Mean tersimpan:", mean_path)
    print("Std tersimpan :", std_path)
    print("Mean:", np.round(linguistic_mean, 4))
    print("Std :", np.round(linguistic_std, 4))


if __name__ == "__main__":
    main()
