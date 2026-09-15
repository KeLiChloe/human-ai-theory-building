"""
Generate OpenAI embeddings for eight theory-text columns:

    race/gender × main-effects/soi × pre-ML/post-ML

Outputs under this folder (one subfolder per combination), e.g.:
    race/main-effects/pre-ML/
    gender/soi/post-ML/

Each folder contains:
    embeddings_wide.parquet
    pca_model_from_3072d.pkl
    run_info.json

Install:
    pip install openai pandas numpy scikit-learn pyarrow tqdm tenacity joblib

Before running:
    export OPENAI_API_KEY="your_api_key_here"

Example:
    python generate_embedding.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.decomposition import PCA
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
# embeddings_openai_text-embedding-3-large → embedding_analysis → textual_analysis → survey-analysis
DEFAULT_INPUT_CSV = SCRIPT_DIR.parents[2] / "anonymous_survey_data.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR

NAME_COLUMN = "Participant ID"
TYPE_COLUMN = "student_0, senior_1, genAI_2"

MODEL = "text-embedding-3-large"

RAW_DIMENSION_FULL_LABEL = 3072
RAW_DIMENSION_FULL_API = None

PCA_SOURCE_DIMENSION = 3072
PCA_VARIANCE_RETAINED = 0.95

TYPE_MAP = {
    "0": "student",
    "1": "senior",
    "2": "GenAI",
}


# Each job: (output subfolder under this embeddings root/, CSV text column)
EMBEDDING_JOBS: list[tuple[str, str]] = [
    ("race/main-effects/pre-ML", "Q Race.4 pre-ML theory (main effects)"),
    ("race/soi/pre-ML", "Q Race.10 pre-ML theory (SOI)"),
    ("race/main-effects/post-ML", "Q Race.12 LLM_refined post-ML theory (main effects)"),
    ("race/soi/post-ML", "Q Race.15 LLM_refined post-ML theory (SOI)"),
    ("gender/main-effects/pre-ML", "Q Gender.4 pre-ML theory (main effects)"),
    ("gender/soi/pre-ML", "Q Gender.10 pre-ML theory (SOI)"),
    ("gender/main-effects/post-ML", "Q Gender.12 LLM_refined post-ML theory (main effects)"),
    ("gender/soi/post-ML", "Q Gender.15 LLM_refined post-ML theory (SOI)"),
]


def parse_output_subdir(output_subdir: str) -> tuple[str, str, str]:
    task, theory_type, phase = output_subdir.split("/", 2)
    return task, theory_type, phase


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def clean_participant_display_name(name: str) -> str:
    """Strip GenAI duplicate suffix, e.g. ``Claude_opus4.7(1)`` → ``Claude_opus4.7``."""
    s = str(name).strip()
    if s.endswith("(1)"):
        return s[:-3]
    return s


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def find_column(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target

    norm_target = " ".join(target.split()).lower()
    for col in df.columns:
        if " ".join(col.split()).lower() == norm_target:
            return col

    related = [
        c for c in df.columns
        if any(token.lower() in c.lower() for token in target.split()[:3])
    ]
    raise KeyError(
        f"Could not find column: {target}\n\n"
        f"Related columns found:\n" + "\n".join(related[:30])
    )


@retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(5))
def embed_batch(
    client: OpenAI,
    texts: List[str],
    api_dimensions: Optional[int],
) -> List[List[float]]:
    kwargs = {
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    if api_dimensions is not None:
        kwargs["dimensions"] = api_dimensions

    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def embed_all_texts(
    client: OpenAI,
    texts: List[str],
    label_dimension: int,
    api_dimensions: Optional[int],
    batch_size: int,
    sleep: float,
    desc: str,
) -> np.ndarray:
    all_embeddings: list[Optional[np.ndarray]] = [None] * len(texts)

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=desc,
    ):
        batch_indices = list(range(start, min(start + batch_size, len(texts))))
        batch_texts = [texts[i] for i in batch_indices]
        batch_embeddings = embed_batch(
            client=client,
            texts=batch_texts,
            api_dimensions=api_dimensions,
        )
        for idx, emb in zip(batch_indices, batch_embeddings):
            all_embeddings[idx] = np.asarray(emb, dtype=np.float32)

        if sleep > 0:
            time.sleep(sleep)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    if embeddings.shape[1] != label_dimension:
        print(
            f"Warning: expected {label_dimension} dimensions, "
            f"but API returned {embeddings.shape[1]} dimensions."
        )
    return embeddings


def fit_pca(raw_embeddings: np.ndarray) -> tuple[PCA, np.ndarray]:
    if raw_embeddings.shape[0] < 2:
        raise ValueError("Need at least two rows to fit PCA.")

    pca = PCA(n_components=PCA_VARIANCE_RETAINED, svd_solver="full")
    pca_embeddings = pca.fit_transform(raw_embeddings).astype(np.float32)
    return pca, pca_embeddings


def vector_to_list(vec: np.ndarray) -> list:
    return vec.astype(float).tolist()


def run_one_job(
    output_subdir: str,
    text_column: str,
    df: pd.DataFrame,
    client: OpenAI,
    output_root: Path,
    batch_size: int,
    sleep: float,
    input_csv: str,
) -> None:
    task, theory_type, phase = parse_output_subdir(output_subdir)
    outdir = output_root / output_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    parquet_path = outdir / "embeddings_wide.parquet"
    pca_model_path = outdir / f"pca_model_from_{PCA_SOURCE_DIMENSION}d.pkl"
    run_info_path = outdir / "run_info.json"

    print("\n" + "=" * 80)
    print(f"Embedding: {output_subdir}")
    print(f"  column: {text_column}")
    print(f"  output: {outdir}")
    print("=" * 80)

    text_col = find_column(df, text_column)
    name_col = find_column(df, NAME_COLUMN)
    type_col = find_column(df, TYPE_COLUMN)

    participant_names = [
        clean_participant_display_name(clean_text(x)) for x in df[name_col]
    ]
    participant_types = [
        TYPE_MAP.get(clean_text(x), clean_text(x) if clean_text(x) else "unknown")
        for x in df[type_col]
    ]
    texts = df[text_col].apply(clean_text).tolist()
    text_word_counts = [word_count(t) for t in texts]

    empty_count = sum(t == "" for t in texts)
    if empty_count > 0:
        raise ValueError(
            f"{output_subdir}: found {empty_count} empty rows in column {text_column}."
        )

    progress_label = output_subdir

    raw_3072 = embed_all_texts(
        client=client,
        texts=texts,
        label_dimension=RAW_DIMENSION_FULL_LABEL,
        api_dimensions=RAW_DIMENSION_FULL_API,
        batch_size=batch_size,
        sleep=sleep,
        desc=f"{progress_label} | 3072d",
    )

    if PCA_SOURCE_DIMENSION != 3072:
        raise ValueError("PCA_SOURCE_DIMENSION must be 3072.")

    print(f"Fitting PCA on {PCA_SOURCE_DIMENSION}d embeddings...")
    pca, pca_embeddings = fit_pca(raw_3072)
    pca_dim = pca_embeddings.shape[1]
    explained = float(np.sum(pca.explained_variance_ratio_))

    output_df = pd.DataFrame({
        "participant_name": participant_names,
        "participant_type": participant_types,
        "text_word_count": text_word_counts,
        "raw_embedding_dimension_3072": [
            vector_to_list(raw_3072[i]) for i in range(raw_3072.shape[0])
        ],
        "pca_embedding": [
            vector_to_list(pca_embeddings[i]) for i in range(pca_embeddings.shape[0])
        ],
    })

    output_df.to_parquet(parquet_path, index=False)
    joblib.dump(pca, pca_model_path)

    run_info = {
        "created_at": datetime.now().isoformat(),
        "task": task,
        "theory_type": theory_type,
        "phase": phase,
        "input_file": input_csv,
        "text_column": text_col,
        "participant_name_column": name_col,
        "participant_type_column": type_col,
        "participant_type_map": TYPE_MAP,
        "embedding_model": MODEL,
        "raw_embedding_dimension_3072": {
            "api_dimensions": RAW_DIMENSION_FULL_API,
            "actual_dimension": int(raw_3072.shape[1]),
        },
        "pca_source_dimension": PCA_SOURCE_DIMENSION,
        "pca_variance_retained_target": PCA_VARIANCE_RETAINED,
        "pca_embedding_dimension": int(pca_dim),
        "pca_explained_variance_retained_actual": explained,
        "n_participants": int(output_df.shape[0]),
        "output_format": "wide",
        "output_columns": list(output_df.columns),
        "output_parquet": str(parquet_path),
        "pca_model": str(pca_model_path),
        "output_subdir": output_subdir,
        "notes": (
            "Each participant appears exactly once. "
            "participant_name is cleaned (whitespace + GenAI duplicate suffix). "
            "text_word_count is the word count of the cleaned theory text. "
            "raw_embedding_dimension_3072 stores the full OpenAI embedding. "
            "pca_embedding is fitted from the 3072d embedding (95% variance retained)."
        ),
    }

    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"Saved parquet: {parquet_path}")
    print(f"Saved PCA model: {pca_model_path}")
    print(f"Saved run info: {run_info_path}")
    print(f"PCA dimensions retained: {pca_dim} ({explained:.4f} variance)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate theory-space embeddings.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help=f"Input CSV path (default: {DEFAULT_INPUT_CSV.name}).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for all embedding sets.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Texts per API call.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between API calls.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    specs = EMBEDDING_JOBS
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV: {args.input}")
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    print(f"Loaded shape: {df.shape}")
    print(f"Jobs to run ({len(specs)}):")
    for output_subdir, text_column in specs:
        print(f"  - {output_subdir} <- {text_column}")

    client = OpenAI()
    for output_subdir, text_column in specs:
        run_one_job(
            output_subdir=output_subdir,
            text_column=text_column,
            df=df,
            client=client,
            output_root=output_root,
            batch_size=args.batch_size,
            sleep=args.sleep,
            input_csv=args.input,
        )

    print("\nDone.")
    print(f"Embedding outputs under: {output_root}")


if __name__ == "__main__":
    main()
