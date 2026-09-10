"""Aplikasi Streamlit: Deteksi Teks Human vs AI.

Menampilkan hasil prediksi dari dua model Hierarchical BERT + LSTM:
  1. Kode ke-1 (BERT + LSTM)
  2. Kode ke-3 (BERT + LSTM + fitur linguistik)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_inference import load_model_bundle, predict_document

WORKSPACE = Path(__file__).resolve().parent
MODEL_DIRS = [
    WORKSPACE / "models" / "kode_ke-1",
    WORKSPACE / "models" / "kode_ke-3",
]

st.set_page_config(
    page_title="Word Detection AI",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Load model (di-cache oleh Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_bundles() -> list:
    bundles = []
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            st.warning(f"Folder model tidak ditemukan: {model_dir}")
            continue
        try:
            bundles.append(load_model_bundle(model_dir))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal memuat model di {model_dir}: {exc}")
    return bundles


# ---------------------------------------------------------------------------
# Helper UI
# ---------------------------------------------------------------------------
def prediction_badge(prediction: str):
    style_map = {
        "AI": ("AI", "red"),
        "Human": ("HUMAN", "green"),
        "Uncertain": ("UNCERTAIN", "orange"),
        "Insufficient Text": ("TEKS TERLALU PENDEK", "gray"),
    }
    label, color = style_map.get(prediction, (prediction, "gray"))
    st.markdown(
        f"<span style='background-color:{color};color:white;"
        f"padding:6px 14px;border-radius:12px;font-weight:700;'>"
        f"{label}</span>",
        unsafe_allow_html=True,
    )


def render_prediction_result(result: dict, bundle) -> None:
    prediction = result.get("prediction", "Uncertain")
    prediction_badge(prediction)

    if prediction == "Insufficient Text":
        st.caption(result.get("reason", ""))
        st.metric("Jumlah kata", result.get("word_count", 0))
        return

    prob_ai = float(result.get("probability_ai", 0.0))
    prob_human = float(result.get("probability_human", 0.0))

    col_left, col_right = st.columns(2)
    with col_left:
        st.metric("Probabilitas AI", f"{prob_ai:.4f}")
        st.progress(min(max(prob_ai, 0.0), 1.0), text="P(AI)")
    with col_right:
        st.metric("Probabilitas Human", f"{prob_human:.4f}")
        st.progress(min(max(prob_human, 0.0), 1.0), text="P(Human)")

    st.caption(result.get("reason", ""))

    details = {
        "Jumlah kata": result.get("word_count"),
        "Chunk dipakai": (
            f"{result.get('selected_chunks')} / "
            f"{result.get('total_available_chunks')} tersedia"
        ),
        "Indeks chunk dipilih": result.get("selected_chunk_indices"),
        "Threshold Human": f"{result.get('threshold_human', 0):.4f}",
        "Threshold AI": f"{result.get('threshold_ai', 0):.4f}",
        "Temperature": f"{result.get('temperature', 1):.4f}",
        "Dimensi fitur linguistik": result.get("linguistic_feature_dim", 0),
    }
    if bundle.linguistic_feature_dim > 0:
        details["Statistik linguistik"] = (
            "dipakai" if result.get("linguistic_stats_used") else "netral (0)"
        )
    st.json(details)


def render_model_config(bundle) -> None:
    with st.expander("Konfigurasi & pipeline"):
        config = dict(bundle.config)
        config["min_words_for_prediction"] = bundle.min_words
        config["force_uncertain_below_words"] = bundle.force_uncertain_below
        st.json(config)


# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🤖 Word Detection AI")
    st.markdown(
        "Deteksi teks **Human vs AI** menggunakan dua model "
        "**Hierarchical BERT + LSTM**. Kedua model dijalankan pada teks yang "
        "sama dan hasil prediksinya ditampilkan berdampingan."
    )

    bundles = get_model_bundles()
    if not bundles:
        st.error("Tidak ada model yang berhasil dimuat.")
        st.stop()

    tab_predict, tab_evaluation = st.tabs(
        ["🔍 Prediksi Teks", "📊 Hasil Evaluasi Model"]
    )

    with tab_predict:
        render_prediction_tab(bundles)

    with tab_evaluation:
        render_evaluation_tab(bundles)


def render_prediction_tab(bundles) -> None:
    st.subheader("Masukkan teks untuk dianalisis")

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    example_text = (
        "Modern communication technologies such as mobile phones, e-mails "
        "and internet chat programs have brought significant changes to our "
        "lives in recent years. Yet, there remains some disagreement as to "
        "whether the overall effect of this innovation has been positive or "
        "negative. Although there are valid arguments to the contrary, it is "
        "my belief that the majority of people in the globe have benefited "
        "greatly from these powerful and effective means of modern "
        "communication. To begin with, mobile phones and other tools of "
        "modern communication facilitate not only contact with friends and "
        "relatives in faraway places but also global business. With the "
        "click of a button, the vast amount of information can be "
        "transmitted from America to China in just a few seconds. "
        "Furthermore, it is generally felt that the access to these tools "
        "of communication is available in every corner of the world. With a "
        "mobile phone or a laptop, a person can talk or send messages "
        "online at a bus stop, in a corner shop or anywhere they could "
        "imagine."
    )

    col_text, col_actions = st.columns([4, 1], vertical_alignment="bottom")
    with col_text:
        input_text = st.text_area(
            "Teks",
            value=st.session_state["input_text"],
            height=260,
            placeholder="Tempel teks/esai/artikel berbahasa Inggris di sini...",
            label_visibility="collapsed",
        )
    with col_actions:
        st.write("")
        use_example = st.button("Gunakan contoh", use_container_width=True)
        clear = st.button("Kosongkan", use_container_width=True)
        predict = st.button(
            "Prediksi", type="primary", use_container_width=True
        )

    if use_example:
        st.session_state["input_text"] = example_text
        st.rerun()
    if clear:
        st.session_state["input_text"] = ""
        st.rerun()

    if not predict:
        st.info(
            "Klik **Prediksi** untuk menjalankan kedua model. "
            "Teks minimal 50 kata (kode ke-1) atau 100 kata (kode ke-3)."
        )
        return

    text = input_text.strip()
    if not text:
        st.warning("Teks masih kosong.")
        return

    with st.spinner("Menjalankan kedua model..."):
        results = []
        for bundle in bundles:
            try:
                results.append((bundle, predict_document(bundle, text)))
            except Exception as exc:  # noqa: BLE001
                st.error(f"{bundle.display_name} gagal memprediksi: {exc}")

    st.divider()
    st.subheader("Hasil Prediksi")

    columns = st.columns(len(results)) if results else []
    for column, (bundle, result) in zip(columns, results):
        with column:
            st.markdown(f"##### {bundle.display_name}")
            render_prediction_result(result, bundle)


def render_evaluation_tab(bundles) -> None:
    st.subheader("Hasil evaluasi yang tersimpan dari training")

    for bundle in bundles:
        model_dir = bundle.model_dir
        with st.container(border=True):
            st.markdown(f"### {bundle.display_name}")
            st.caption(f"Notebook: {bundle.notebook_name or 'lihat pipeline'}")

            metrics_path = model_dir / "test_metrics.json"
            history_path = model_dir / "training_history.csv"
            predictions_path = model_dir / "test_predictions.csv"
            image_path = model_dir / "hierarchical_bert_lstm.png"

            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as file:
                    metrics = json.load(file)

                metric_cols = st.columns(5)
                for col, key in zip(
                    metric_cols,
                    ["accuracy", "precision", "recall", "f1", "roc_auc"],
                ):
                    col.metric(key, f"{metrics.get(key, float('nan')):.4f}")
                metric_cols2 = st.columns(3)
                for col, key in zip(
                    metric_cols2,
                    ["coverage", "uncertain_rate", "selective_accuracy"],
                ):
                    col.metric(key, f"{metrics.get(key, float('nan')):.4f}")

            left, right = st.columns(2)
            with left:
                if history_path.exists():
                    history = pd.read_csv(history_path)
                    st.markdown("**Training history**")
                    st.line_chart(
                        history.set_index("epoch")[
                            ["train_loss", "val_loss"]
                        ]
                    )
                    st.line_chart(
                        history.set_index("epoch")[
                            ["val_accuracy", "val_f1", "val_roc_auc"]
                        ]
                    )
            with right:
                if image_path.exists():
                    st.markdown("**Arsitektur model**")
                    st.image(str(image_path), use_container_width=True)

            if predictions_path.exists():
                st.markdown("**Test predictions (10 baris pertama)**")
                predictions = pd.read_csv(predictions_path)
                st.dataframe(predictions.head(10), use_container_width=True)

            render_model_config(bundle)


main()
