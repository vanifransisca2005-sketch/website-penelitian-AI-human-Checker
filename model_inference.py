"""Inference helpers untuk model Hierarchical BERT + LSTM.

Modul ini mengikuti pipeline dari dua notebook:

1. ``skripsi-1.ipynb``           -> model "Kode ke-1" (BERT + LSTM, tanpa fitur linguistik).
2. ``banding-new-skripsi (2).ipynb`` -> model "Kode ke-3" (BERT + LSTM + fitur linguistik).

Fungsi utama:
    load_model_bundle(model_dir)  -> ModelBundle
    predict_document(bundle, text) -> dict hasil prediksi
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertConfig, BertModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Konstanta BERT base-uncased (dibangun manual supaya tidak perlu download)
# ---------------------------------------------------------------------------
BERT_BASE_UNCASED_CONFIG = dict(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    position_embedding_type="absolute",
    use_cache=True,
    classifier_dropout=None,
)


# ---------------------------------------------------------------------------
# Preprocessing teks (sama persis dengan notebook)
# ---------------------------------------------------------------------------
def minimal_clean_text(text: str) -> str:
    """Cleaning konservatif: hanya normalisasi Unicode/kontrol/spasi."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        character
        for character in text
        if character in "\n\t"
        or not unicodedata.category(character).startswith("C")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def select_evenly_spaced_indices(total_items: int, max_items: int) -> list[int]:
    if total_items <= max_items:
        return list(range(total_items))
    selected = np.linspace(0, total_items - 1, num=max_items)
    selected = np.round(selected).astype(int)
    return list(dict.fromkeys(selected.tolist()))


# ---------------------------------------------------------------------------
# Hierarchical chunk encoding (sama dengan notebook banding-new-skripsi)
# ---------------------------------------------------------------------------
def build_inputs_with_special_tokens(
    tokenizer: Any,
    token_ids_0: list[int],
) -> list[int]:
    """Membangun sequence dengan special tokens [CLS] ... [SEP].

    ``PreTrainedTokenizer.build_inputs_with_special_tokens`` tersedia di
    transformers v4, tetapi dihapus di transformers v5. Fallback manual ini
    membuat kode bekerja di kedua versi.
    """
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return list(tokenizer.build_inputs_with_special_tokens(token_ids_0))

    sequence: list[int] = []
    if tokenizer.cls_token_id is not None:
        sequence.append(int(tokenizer.cls_token_id))
    sequence.extend(int(token_id) for token_id in token_ids_0)
    if tokenizer.sep_token_id is not None:
        sequence.append(int(tokenizer.sep_token_id))
    return sequence


def encode_document_chunks(
    text: str,
    tokenizer: Any,
    chunk_length: int = 256,
    stride: int = 192,
    max_chunks: int = 4,
) -> dict[str, Any]:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    if not token_ids:
        token_ids = [tokenizer.unk_token_id]

    usable_length = chunk_length - 2
    all_chunks: list[list[int]] = []

    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start : start + usable_length]
        if not chunk:
            break
        all_chunks.append(chunk)
        if start + usable_length >= len(token_ids):
            break

    selected_indices = select_evenly_spaced_indices(len(all_chunks), max_chunks)
    selected_chunks = [all_chunks[index] for index in selected_indices]

    input_ids = np.full(
        (max_chunks, chunk_length), tokenizer.pad_token_id, dtype=np.int64
    )
    attention_mask = np.zeros((max_chunks, chunk_length), dtype=np.int64)
    chunk_mask = np.zeros(max_chunks, dtype=np.int64)

    for chunk_index, chunk in enumerate(selected_chunks):
        sequence = build_inputs_with_special_tokens(
            tokenizer, list(chunk[: chunk_length - 2])
        )
        if len(sequence) > chunk_length or any(
            token_id is None for token_id in sequence
        ):
            raise ValueError(
                "Special token tokenizer tidak valid untuk CHUNK_LENGTH."
            )
        sequence = [int(token_id) for token_id in sequence]
        sequence_length = len(sequence)
        input_ids[chunk_index, :sequence_length] = sequence
        attention_mask[chunk_index, :sequence_length] = 1
        chunk_mask[chunk_index] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "total_chunks": len(all_chunks),
        "selected_indices": selected_indices,
    }


# ---------------------------------------------------------------------------
# Arsitektur model (sama dengan notebook)
# ---------------------------------------------------------------------------
class HierarchicalBertEncoder(nn.Module):
    def __init__(
        self,
        bert_model: nn.Module,
        pooling: str = "mean",
        special_ids_to_exclude: set[int] | None = None,
    ):
        super().__init__()
        self.bert = bert_model
        self.pooling = pooling
        self.hidden_size = self.bert.config.hidden_size
        self.special_ids_to_exclude = special_ids_to_exclude or set()

    def pool_chunk_tokens(
        self,
        token_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return token_embeddings[:, 0]

        if self.pooling != "mean":
            raise ValueError("Pooling harus 'mean' atau 'cls'.")

        content_mask = attention_mask.bool()
        for special_token_id in self.special_ids_to_exclude:
            if special_token_id is not None:
                content_mask = content_mask & input_ids.ne(int(special_token_id))

        content_mask_float = content_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * content_mask_float).sum(dim=1) / content_mask_float.sum(
            dim=1
        ).clamp(min=1.0)
        return pooled

    def encode_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_chunks, sequence_length = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * max_chunks, sequence_length)
        flat_attention_mask = attention_mask.view(
            batch_size * max_chunks, sequence_length
        )
        flat_chunk_mask = chunk_mask.view(-1).bool()

        valid_input_ids = flat_input_ids[flat_chunk_mask]
        valid_attention_mask = flat_attention_mask[flat_chunk_mask]

        bert_output = self.bert(
            input_ids=valid_input_ids,
            attention_mask=valid_attention_mask,
        )

        valid_chunk_vectors = self.pool_chunk_tokens(
            token_embeddings=bert_output.last_hidden_state,
            input_ids=valid_input_ids,
            attention_mask=valid_attention_mask,
        )

        all_chunk_vectors = torch.zeros(
            (batch_size * max_chunks, self.hidden_size),
            dtype=valid_chunk_vectors.dtype,
            device=input_ids.device,
        )
        all_chunk_vectors[flat_chunk_mask] = valid_chunk_vectors
        return all_chunk_vectors.view(batch_size, max_chunks, self.hidden_size)


class HierarchicalBertLSTMClassifier(nn.Module):
    def __init__(
        self,
        bert_model: nn.Module,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.40,
        chunk_pooling: str = "mean",
        linguistic_feature_dim: int = 0,
        special_ids_to_exclude: set[int] | None = None,
    ):
        super().__init__()
        self.encoder = HierarchicalBertEncoder(
            bert_model,
            pooling=chunk_pooling,
            special_ids_to_exclude=special_ids_to_exclude,
        )
        self.bert = self.encoder.bert
        self.lstm = nn.LSTM(
            self.encoder.hidden_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.linguistic_feature_dim = int(linguistic_feature_dim)
        self.classifier = nn.Linear(
            lstm_hidden_size + self.linguistic_feature_dim, 1
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        linguistic_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chunk_vectors = self.encoder.encode_chunks(
            input_ids, attention_mask, chunk_mask
        )
        chunk_lengths = chunk_mask.sum(dim=1).clamp(min=1).to("cpu")
        packed_chunks = pack_padded_sequence(
            chunk_vectors,
            chunk_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_state, _) = self.lstm(packed_chunks)
        document_semantic = hidden_state[-1]

        if self.linguistic_feature_dim > 0:
            if linguistic_features is None:
                raise ValueError(
                    "Model ini membutuhkan linguistic_features."
                )
            combined = torch.cat(
                [document_semantic, linguistic_features], dim=1
            )
        else:
            combined = document_semantic

        return self.classifier(self.dropout(combined)).squeeze(-1)


# ---------------------------------------------------------------------------
# Calibration & decision (sama dengan notebook)
# ---------------------------------------------------------------------------
def sigmoid_numpy(logits: np.ndarray | float) -> np.ndarray | float:
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def calibrated_probabilities(logits: Any, temperature: float) -> np.ndarray:
    return sigmoid_numpy(np.asarray(logits) / max(float(temperature), 1e-6))


def three_way_decision(
    probability_ai: float,
    threshold_human: float,
    threshold_ai: float,
    word_count: int,
    min_words: int,
    force_uncertain_below: int,
) -> tuple[str, str]:
    """Mengembalikan (prediction, reason) mengikuti pipeline notebook."""
    if word_count < min_words:
        return (
            "Insufficient Text",
            "Teks terlalu pendek untuk prediksi yang layak.",
        )
    if word_count < force_uncertain_below:
        return (
            "Uncertain",
            "Teks cukup untuk dianalisis, tetapi masih terlalu pendek untuk keputusan tegas.",
        )
    if probability_ai <= threshold_human:
        return "Human", "Probabilitas berada di zona Human."
    if probability_ai >= threshold_ai:
        return "AI", "Probabilitas berada di zona AI."
    return "Uncertain", "Probabilitas berada di zona ketidakpastian."


# ---------------------------------------------------------------------------
# Fitur linguistik (khusus model kode ke-3)
# ---------------------------------------------------------------------------
POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX",
    "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN",
    "SCONJ", "SYM", "VERB", "X",
]


def _load_spacy_nlp():
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError as error:
        raise RuntimeError(
            "Pipeline spaCy 'en_core_web_sm' belum terpasang. "
            "Jalankan '.\\venv\\Scripts\\python.exe -m spacy download "
            "en_core_web_sm' atau instal ulang requirements.txt."
        ) from error
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@torch.no_grad()
def compute_perplexity_burstiness(
    text: str,
    tokenizer: Any,
    model: Any,
    max_tokens: int = 512,
) -> tuple[float, float]:
    """Mengembalikan (log_perplexity, burstiness) seperti notebook."""
    if not text or not text.strip():
        return 0.0, 0.0

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"]

    if input_ids.shape[1] < 2:
        return 0.0, 0.0

    outputs = model(input_ids)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )
    token_nll = token_nll.detach().cpu().numpy()

    mean_nll = float(token_nll.mean())
    std_nll = float(token_nll.std())

    doc_perplexity = float(np.exp(min(mean_nll, 20.0)))
    burstiness = std_nll / (mean_nll + 1e-6)
    return float(np.log1p(doc_perplexity)), float(burstiness)


def extract_linguistic_features(
    text: str,
    nlp: Any,
    perplexity_tokenizer: Any | None = None,
    perplexity_model: Any | None = None,
    enable_perplexity: bool = True,
) -> np.ndarray:
    """Mengembalikan vektor 30 fitur (14 stylometric + 16 POS)."""
    text = "" if text is None else str(text)
    doc = nlp(text[:20000])

    tokens = [token for token in doc if not token.is_space]
    words = [token for token in tokens if token.is_alpha]
    word_texts = [token.text.lower() for token in words]

    sentences = [
        sentence for sentence in doc.sents if sentence.text.strip()
    ]

    lengths = np.asarray(
        [
            len([token for token in sentence if token.is_alpha])
            for sentence in sentences
        ],
        dtype=np.float32,
    )

    word_count = max(len(words), 1)
    char_count = max(len(text), 1)

    import pandas as pd

    counts = (
        pd.Series(word_texts).value_counts()
        if word_texts
        else pd.Series(dtype=float)
    )

    pos_features = np.asarray(
        [
            sum(token.pos_ == tag for token in words) / word_count
            for tag in POS_TAGS
        ],
        dtype=np.float32,
    )

    if enable_perplexity and perplexity_tokenizer is not None and perplexity_model is not None:
        log_perplexity, burstiness = compute_perplexity_burstiness(
            text,
            tokenizer=perplexity_tokenizer,
            model=perplexity_model,
        )
    else:
        log_perplexity, burstiness = 0.0, 0.0

    style_features = np.asarray(
        [
            np.log1p(len(words)),
            np.log1p(len(sentences)),
            float(lengths.mean() if len(lengths) > 0 else 0),
            float(lengths.std() if len(lengths) > 0 else 0),
            len(set(word_texts)) / word_count,
            float((counts == 1).sum() / word_count),
            text.count(",") / char_count,
            text.count(";") / char_count,
            text.count(":") / char_count,
            sum(character in "!?" for character in text) / char_count,
            sum(character.isupper() for character in text) / char_count,
            text.count("\n\n") + 1,
            log_perplexity,
            burstiness,
        ],
        dtype=np.float32,
    )

    return np.concatenate([style_features, pos_features]).astype(np.float32)


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    key: str
    display_name: str
    notebook_name: str
    model_dir: Path
    config: dict[str, Any]
    tokenizer: Any
    model: HierarchicalBertLSTMClassifier
    device: torch.device
    linguistic_feature_dim: int = 0
    linguistic_mean: np.ndarray | None = None
    linguistic_std: np.ndarray | None = None
    spacy_nlp: Any | None = None
    perplexity_tokenizer: Any | None = None
    perplexity_model: Any | None = None
    min_words: int = 50
    force_uncertain_below: int = 80
    threshold_human: float = 0.5
    threshold_ai: float = 0.5
    temperature: float = 1.0
    chunk_length: int = 256
    chunk_stride: int = 192
    max_chunks: int = 4
    text_mode: str = "clean"

    @property
    def has_linguistic_stats(self) -> bool:
        return self.linguistic_mean is not None and self.linguistic_std is not None


def _build_bert_model() -> BertModel:
    config = BertConfig(**BERT_BASE_UNCASED_CONFIG)
    return BertModel(config)


def load_model_bundle(model_dir: str | Path) -> ModelBundle:
    """Memuat satu model + config + tokenizer dari folder hasil training."""
    model_dir = Path(model_dir)
    config_path = model_dir / "experiment_config.json"
    checkpoint_path = model_dir / "hierarchical_bert_lstm_best.pt"
    tokenizer_path = model_dir / "tokenizer"

    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model tidak lengkap di {model_dir}. "
            "Butuh experiment_config.json dan hierarchical_bert_lstm_best.pt."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model_state_dict"]

    linguistic_feature_dim = int(
        checkpoint.get("linguistic_feature_dim", 0)
    )

    # Kode ke-1 dan ke-3 sama-sama arsitektur BERT+LSTM; bedanya hanya
    # ada/tidaknya fitur linguistik. Untuk pooling, kode ke-1 hanya
    # mengecualikan CLS/SEP/PAD, sedangkan kode ke-3 mengecualikan semua
    # special token (mengikuti notebook masing-masing).
    if linguistic_feature_dim > 0:
        special_ids_to_exclude = set(
            int(token_id)
            for token_id in tokenizer.all_special_ids
            if token_id is not None
        )
    else:
        special_ids_to_exclude = {
            int(token_id)
            for token_id in (
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )
            if token_id is not None
        }

    bert_model = _build_bert_model()
    model = HierarchicalBertLSTMClassifier(
        bert_model=bert_model,
        lstm_hidden_size=int(checkpoint.get("lstm_hidden_size", 128)),
        lstm_num_layers=int(checkpoint.get("lstm_num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.40)),
        chunk_pooling=checkpoint.get("chunk_pooling", "mean"),
        linguistic_feature_dim=linguistic_feature_dim,
        special_ids_to_exclude=special_ids_to_exclude,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Konfigurasi notebook:
    # - Kode ke-1 (near_duplicate_grouping=false, target_fpr=0.01):
    #     MIN_WORDS_FOR_PREDICTION=50, FORCE_UNCERTAIN_BELOW_WORDS=80
    # - Kode ke-3 (near_duplicate_grouping=true, target_fpr=0.02):
    #     MIN_WORDS_FOR_PREDICTION=100, FORCE_UNCERTAIN_BELOW_WORDS=120
    if config.get("near_duplicate_grouping") is True:
        min_words = 100
        force_uncertain_below = 120
    else:
        min_words = 50
        force_uncertain_below = 80

    display_names = {
        "kode_ke-1": "Kode ke-1 (BERT + LSTM)",
        "kode_ke-3": "Kode ke-3 (BERT + LSTM + Linguistik)",
        "model1": "Kode ke-1 (BERT + LSTM)",
        "model2": "Kode ke-3 (BERT + LSTM + Linguistik)",
    }
    notebook_names = {
        "kode_ke-1": "skripsi-1.ipynb",
        "kode_ke-3": "banding-new-skripsi (2).ipynb",
        "model1": "skripsi-1.ipynb",
        "model2": "banding-new-skripsi (2).ipynb",
    }

    bundle = ModelBundle(
        key=model_dir.name,
        display_name=display_names.get(model_dir.name, model_dir.name),
        notebook_name=notebook_names.get(model_dir.name, ""),
        model_dir=model_dir,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        linguistic_feature_dim=linguistic_feature_dim,
        min_words=min_words,
        force_uncertain_below=force_uncertain_below,
        threshold_human=float(config.get("threshold_human", 0.5)),
        threshold_ai=float(config.get("threshold_ai", 0.5)),
        temperature=float(config.get("temperature", 1.0)),
        chunk_length=int(config.get("chunk_length", 256)),
        chunk_stride=int(config.get("chunk_stride", 192)),
        max_chunks=int(config.get("max_chunks", 4)),
        text_mode=config.get("text_mode", "clean"),
    )

    # Model kode ke-3: siapkan fitur linguistik bila statistik tersedia.
    if linguistic_feature_dim > 0:
        mean_path = model_dir / "linguistic_mean.npy"
        std_path = model_dir / "linguistic_std.npy"
        if mean_path.exists() and std_path.exists():
            bundle.linguistic_mean = np.load(mean_path).astype(np.float32)
            bundle.linguistic_std = np.load(std_path).astype(np.float32)
        elif (
            "linguistic_mean" in checkpoint
            and "linguistic_std" in checkpoint
        ):
            # Fallback: notebook terbaru juga menyimpan statistik ini
            # di dalam checkpoint .pt.
            bundle.linguistic_mean = np.asarray(
                checkpoint["linguistic_mean"], dtype=np.float32
            )
            bundle.linguistic_std = np.asarray(
                checkpoint["linguistic_std"], dtype=np.float32
            )
        else:
            bundle.linguistic_mean = None
            bundle.linguistic_std = None

    return bundle


def _ensure_linguistic_components(bundle: ModelBundle) -> None:
    """Memuat spaCy + distilgpt2 hanya saat diperlukan (lazy)."""
    if bundle.linguistic_feature_dim <= 0:
        return
    if bundle.spacy_nlp is None:
        bundle.spacy_nlp = _load_spacy_nlp()
    if bundle.perplexity_tokenizer is None or bundle.perplexity_model is None:
        from transformers import AutoModelForCausalLM

        def _load_distilgpt2():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "distilgpt2", local_files_only=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    "distilgpt2", local_files_only=True
                )
                return tokenizer, model
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                return tokenizer, model

        bundle.perplexity_tokenizer, bundle.perplexity_model = _load_distilgpt2()
        if bundle.perplexity_tokenizer.pad_token is None:
            bundle.perplexity_tokenizer.pad_token = (
                bundle.perplexity_tokenizer.eos_token
            )
        bundle.perplexity_model.to(bundle.device)
        bundle.perplexity_model.eval()
        for parameter in bundle.perplexity_model.parameters():
            parameter.requires_grad = False


@torch.no_grad()
def predict_document(bundle: ModelBundle, text: str) -> dict[str, Any]:
    """Menjalankan pipeline prediksi persis seperti notebook untuk satu teks."""
    if not isinstance(text, str):
        raise TypeError("Input harus berupa string.")

    cleaned_text = minimal_clean_text(text)
    word_count = len(cleaned_text.split())

    base_result: dict[str, Any] = {
        "model": bundle.display_name,
        "word_count": word_count,
    }

    # Pipeline notebook: teks di bawah MIN_WORDS langsung "Insufficient Text".
    if word_count < bundle.min_words:
        base_result["prediction"] = "Insufficient Text"
        base_result["reason"] = "Teks terlalu pendek untuk prediksi yang layak."
        return base_result

    model_text = cleaned_text if bundle.text_mode == "clean" else text.strip()

    encoded = encode_document_chunks(
        text=model_text,
        tokenizer=bundle.tokenizer,
        chunk_length=bundle.chunk_length,
        stride=bundle.chunk_stride,
        max_chunks=bundle.max_chunks,
    )

    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=bundle.device
    ).unsqueeze(0)
    attention_mask = torch.tensor(
        encoded["attention_mask"], dtype=torch.long, device=bundle.device
    ).unsqueeze(0)
    chunk_mask = torch.tensor(
        encoded["chunk_mask"], dtype=torch.long, device=bundle.device
    ).unsqueeze(0)

    linguistic_features = None
    if bundle.linguistic_feature_dim > 0:
        if bundle.has_linguistic_stats:
            _ensure_linguistic_components(bundle)
            features = extract_linguistic_features(
                text,
                nlp=bundle.spacy_nlp,
                perplexity_tokenizer=bundle.perplexity_tokenizer,
                perplexity_model=bundle.perplexity_model,
                enable_perplexity=True,
            ).astype(np.float32)
            features = (
                features - bundle.linguistic_mean
            ) / bundle.linguistic_std
        else:
            # Fallback: rata-rata fitur terstandarisasi = 0 (kontribusi
            # linguistik netral). Terjadi hanya bila linguistic_mean/std
            # belum di-generate.
            features = np.zeros(
                bundle.linguistic_feature_dim, dtype=np.float32
            )

        linguistic_features = torch.tensor(
            features, dtype=torch.float32, device=bundle.device
        ).unsqueeze(0)

    logit = bundle.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        chunk_mask=chunk_mask,
        linguistic_features=linguistic_features,
    )

    probability_ai = float(
        calibrated_probabilities(
            logit.detach().cpu().numpy(),
            bundle.temperature,
        )[0]
    )

    prediction, reason = three_way_decision(
        probability_ai=probability_ai,
        threshold_human=bundle.threshold_human,
        threshold_ai=bundle.threshold_ai,
        word_count=word_count,
        min_words=bundle.min_words,
        force_uncertain_below=bundle.force_uncertain_below,
    )

    return {
        **base_result,
        "prediction": prediction,
        "probability_human": 1.0 - probability_ai,
        "probability_ai": probability_ai,
        "selected_chunks": int(encoded["chunk_mask"].sum()),
        "total_available_chunks": int(encoded["total_chunks"]),
        "selected_chunk_indices": encoded["selected_indices"],
        "threshold_human": bundle.threshold_human,
        "threshold_ai": bundle.threshold_ai,
        "temperature": bundle.temperature,
        "linguistic_feature_dim": int(bundle.linguistic_feature_dim),
        "linguistic_stats_used": bundle.has_linguistic_stats,
        "reason": reason,
    }
