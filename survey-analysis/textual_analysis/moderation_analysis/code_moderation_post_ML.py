"""Code LLM-refined post-ML SOI theories for moderation mechanisms.

Uses only Q *.15 LLM_refined post-ML theory (SOI). Do not use the raw
Q *.15 post-ML theory (SOI) text.

Each theory is coded against exactly six pairs:
USER_SELECTED_rank_1–3 = the respondent's three selected SOI pairs;
ML_rank_1–3 = the ML top-3 pairs. Duplicate feature combinations across the two
lists are kept as separate coding units.

One API call = one respondent x one task. After each successful call the JSON
file is replaced atomically, so an interrupted run keeps every theory already coded.

Requires OPENAI_API_KEY.

    python code_moderation_post_ML.py
    python code_moderation_post_ML.py --limit 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "moderation_coding_post_ML.md"
CSV_PATH = HERE / "moderation_analysis_data.csv"
ML_PATH = HERE.parent.parent / "forecasts" / "ML_results_second_order_interactions.json"
OUTPUT_PATH = HERE / "moderation_coding_post_ML.json"

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING_EFFORT = "xhigh"
CODING_SCHEMA = "user3_ml3_v2"
PAIR_IDS = (
    "USER_SELECTED_rank_1",
    "USER_SELECTED_rank_2",
    "USER_SELECTED_rank_3",
    "ML_rank_1",
    "ML_rank_2",
    "ML_rank_3",
)
N_PAIRS = len(PAIR_IDS)

FEATURE_DEFINITIONS = {
    "social_science": "the paper is published in a social science journal",
    "natural_science": "the paper is published in a natural science journal",
    "engineering_and_technology": "the paper is published in an engineering and technology journal",
    "num_authors": "the number of authors",
    "female": "the share of authors who are female",
    "asian": "the share of authors who are Asian",
    "black": "the share of authors who are Black",
    "hispanic_and_other": "the share of authors who are Hispanic or in other racial categories",
    "white": "the share of authors who are White",
    "authors_race_diversity_score": "the racial diversity of the author team",
    "country_race_diversity_score": "the racial diversity of the authors' countries",
    "news_inequality_mentions_3_years": "inequality mentions in the news over the previous three years",
    "paper_inequality_mentions_3_years": "inequality mentions in academic papers over the previous three years",
}

TASKS = (
    {
        "task": "racial_inequality",
        "ml_key": "race",
        "research_domain": "Racial inequality",
        "outcome_name": "discussing racial inequality",
        "outcome_definition": "whether a paper discusses racial inequality",
        "pair_cols": (
            "Q Race.6 (SOI, 1st)",
            "Q Race.7 (SOI, 2nd)",
            "Q Race.8 (SOI, 3rd)",
        ),
        "sign_cols": (
            "Q Race.9 (SOI, sign, 1st)",
            "Q Race.9 (SOI, sign, 2nd)",
            "Q Race.9 (SOI, sign, 3rd)",
        ),
        "theory_col": "Q Race.15 LLM_refined post-ML theory (SOI)",
    },
    {
        "task": "gender_inequality",
        "ml_key": "gender",
        "research_domain": "Gender inequality",
        "outcome_name": "discussing gender inequality",
        "outcome_definition": "whether a paper discusses gender inequality",
        "pair_cols": (
            "Q Gender.6 (SOI, 1st)",
            "Q Gender.7 (SOI, 2nd)",
            "Q Gender.8 (SOI, 3rd)",
        ),
        "sign_cols": (
            "Q Gender.9 (SOI, sign, 1st)",
            "Q Gender.9 (SOI, sign, 2nd)",
            "Q Gender.9 (SOI, sign, 3rd)",
        ),
        "theory_col": "Q Gender.15 LLM_refined post-ML theory (SOI)",
    },
)

CLASSIFICATIONS = {
    "terminal_curvature",
    "nonterminal_upstream_interaction",
    "mixed_terminal",
    "restatement_only",
    "not_addressed",
    "unclear",
}

PAIR_FIELDS = (
    "features",
    "addressed",
    "curvature",
    "upstream_interaction",
    "primary_classification",
    "interaction_sign",
    "key_evidence",
)
NESTED_FIELDS = {
    "curvature": ("present", "mechanism", "evidence"),
    "upstream_interaction": ("present", "mechanism", "evidence"),
    "interaction_sign": ("mechanism_implied_sign", "consistent_with_sign"),
}

SOURCE_LABEL = {
    "user_selected": "USER_SELECTED",
    "ml_selected": "ML",
}


def load_prompt_template() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    text = re.sub(r"\A<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)
    return text.strip() + "\n"


def load_ml_top3() -> dict[str, list[dict[str, object]]]:
    data = json.loads(ML_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, object]]] = {}
    for task in TASKS:
        rows = sorted(data[task["ml_key"]], key=lambda row: int(row["rank"]))
        if len(rows) < 3:
            raise ValueError(f"{ML_PATH.name} {task['ml_key']} needs at least 3 pairs")
        pairs = []
        for row in rows[:3]:
            feature_a = row["feature_1"]
            feature_b = row["feature_2"]
            sign = str(row["sign"]).strip()
            ml_rank = int(row["rank"])
            if feature_a not in FEATURE_DEFINITIONS or feature_b not in FEATURE_DEFINITIONS:
                raise ValueError(f"Unknown ML feature in {task['ml_key']}: {feature_a}, {feature_b}")
            if sign not in {"+", "-"}:
                raise ValueError(f"Bad ML sign for {task['ml_key']} rank {ml_rank}: {sign!r}")
            pairs.append({
                "ml_rank": ml_rank,
                "feature_a": feature_a,
                "feature_b": feature_b,
                "ml_sign": sign,
            })
        got_ranks = [int(pair["ml_rank"]) for pair in pairs]
        if got_ranks != [1, 2, 3]:
            raise ValueError(f"{ML_PATH.name} {task['ml_key']} ranks must be 1..3, got {got_ranks}")
        out[task["task"]] = pairs
    return out


def required_columns() -> list[str]:
    cols = ["Participant ID", "student_0, senior_1, genAI_2"]
    for task in TASKS:
        cols.extend(task["pair_cols"])
        cols.extend(task["sign_cols"])
        cols.append(task["theory_col"])
    return cols


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in required_columns() if col not in fieldnames]
        if missing:
            listed = "\n".join(f"  {col}" for col in missing)
            raise ValueError(f"CSV is missing required columns:\n{listed}")
        rows = list(reader)
    ids = [row["Participant ID"].strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Participant ID values are not unique")
    return rows


def parse_pair(cell: str) -> tuple[str, str]:
    parts = [part.strip() for part in cell.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected feature_a,feature_b, got {cell!r}")
    unknown = [part for part in parts if part not in FEATURE_DEFINITIONS]
    if unknown:
        raise ValueError(f"Unknown feature token(s): {unknown}")
    return parts[0], parts[1]


def format_pairs_to_code(pairs: list[dict[str, object]], outcome_name: str) -> str:
    blocks = []
    for pair in pairs:
        feature_a = str(pair["feature_a"])
        feature_b = str(pair["feature_b"])
        if pair["source"] == "user_selected":
            sign_line = f"   predicted_sign: {pair['user_predicted_sign']}"
        else:
            sign_line = f"   ml_sign: {pair['ml_sign']}"
        blocks.append(
            f"{pair['pair_id']}. {feature_a} × {feature_b} → {outcome_name}\n"
            f"   {feature_a}: {FEATURE_DEFINITIONS[feature_a]}\n"
            f"   {feature_b}: {FEATURE_DEFINITIONS[feature_b]}\n"
            f"   Source: {SOURCE_LABEL[str(pair['source'])]}\n"
            f"{sign_line}"
        )
    return "\n\n".join(blocks)


def fill_prompt(
    template: str,
    task: dict,
    pairs_to_code: list[dict[str, object]],
    theory: str,
) -> str:
    values = {
        "RESEARCH_DOMAIN": task["research_domain"],
        "OUTCOME_NAME": task["outcome_name"],
        "OUTCOME_DEFINITION": task["outcome_definition"],
        "THEORY_TEXT": theory,
        "PAIRS_TO_CODE_BLOCK": format_pairs_to_code(pairs_to_code, task["outcome_name"]),
    }

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ValueError(f"Unfilled placeholder: {key}")
        return values[key]

    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", replace, template)


def row_user_pairs(row: dict[str, str], task: dict) -> list[dict[str, object]]:
    pairs = []
    for rank, (pair_col, sign_col) in enumerate(zip(task["pair_cols"], task["sign_cols"]), start=1):
        feature_a, feature_b = parse_pair(row[pair_col])
        sign = row[sign_col].strip()
        if sign not in {"+", "-"}:
            raise ValueError(f"{row['Participant ID'].strip()} {task['task']} rank {rank}: sign {sign!r}")
        pairs.append({
            "user_rank": rank,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "predicted_sign": sign,
        })
    return pairs


def build_pairs_to_code(
    user_pairs: list[dict[str, object]],
    ml_pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Always 6 pairs: USER_SELECTED_rank_1–3, then ML_rank_1–3. Keep duplicates."""
    if len(user_pairs) != 3 or len(ml_pairs) != 3:
        raise ValueError("Need exactly 3 user pairs and 3 ML pairs")
    pairs_to_code: list[dict[str, object]] = []
    for pair in user_pairs:
        user_rank = int(pair["user_rank"])
        pairs_to_code.append({
            "pair_id": f"USER_SELECTED_rank_{user_rank}",
            "feature_a": pair["feature_a"],
            "feature_b": pair["feature_b"],
            "source": "user_selected",
            "user_rank": user_rank,
            "ml_rank": None,
            "user_predicted_sign": pair["predicted_sign"],
            "ml_sign": None,
        })
    for pair in ml_pairs:
        ml_rank = int(pair["ml_rank"])
        pairs_to_code.append({
            "pair_id": f"ML_rank_{ml_rank}",
            "feature_a": pair["feature_a"],
            "feature_b": pair["feature_b"],
            "source": "ml_selected",
            "user_rank": None,
            "ml_rank": ml_rank,
            "user_predicted_sign": None,
            "ml_sign": pair["ml_sign"],
        })
    got_ids = [str(pair["pair_id"]) for pair in pairs_to_code]
    if got_ids != list(PAIR_IDS):
        raise ValueError(f"Expected pair_ids {list(PAIR_IDS)}, got {got_ids}")
    return pairs_to_code


def load_output(path: Path, model: str, reasoning_effort: str) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ValueError(f"{path} must be a JSON object with a records list")
        stored_model = data.get("model")
        stored_effort = data.get("reasoning_effort")
        stored_schema = data.get("coding_schema")
        if stored_schema and stored_schema != CODING_SCHEMA:
            raise ValueError(
                f"{path.name} has coding_schema={stored_schema!r}, but this script "
                f"writes {CODING_SCHEMA!r}. Rename/move the old file or pass --output."
            )
        if data.get("records") and not stored_schema:
            raise ValueError(
                f"{path.name} looks like an older run (no coding_schema). "
                f"Rename/move it or pass --output for the {CODING_SCHEMA} recode."
            )
        if stored_model != model or stored_effort != reasoning_effort:
            raise ValueError(
                f"{path.name} was started with model={stored_model!r}, "
                f"reasoning_effort={stored_effort!r}. This run is model={model!r}, "
                f"reasoning_effort={reasoning_effort!r}. Use the same settings, or pass a new --output."
            )
        return data
    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "coding_schema": CODING_SCHEMA,
        "prompt_file": PROMPT_PATH.name,
        "source_csv": CSV_PATH.name,
        "ml_file": ML_PATH.name,
        "records": [],
    }


def completed_keys(data: dict) -> set[tuple[str, str]]:
    done = set()
    for record in data["records"]:
        if record.get("coding") is None:
            continue
        if record.get("coding_schema", data.get("coding_schema")) != CODING_SCHEMA:
            continue
        done.add((record["participant_id"], record["task"]))
    return done


def save_output(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["coding_schema"] = CODING_SCHEMA
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_pair_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Bad pair_id: {value!r}")
    pair_id = value.strip()
    if pair_id not in PAIR_IDS:
        raise ValueError(f"Unknown pair_id: {pair_id!r}")
    return pair_id


def require_fields(obj: object, fields: tuple[str, ...], label: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{label} is not an object")
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ValueError(f"{label} is missing {missing}")
    return obj


def extract_json(text: str, expected_pairs: list[dict[str, object]]) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        fenced = re.search(r"\A```(?:json)?\s*(.*?)```\s*\Z", raw, flags=re.DOTALL)
        if not fenced:
            raise ValueError("Model output has an unclosed code fence")
        raw = fenced.group(1).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object")
    reconstruction = require_fields(
        parsed.get("causal_reconstruction"),
        ("summary", "structure"),
        "causal_reconstruction",
    )
    if not isinstance(reconstruction["structure"], list):
        raise ValueError("causal_reconstruction.structure must be a list")
    if "confidence" not in parsed:
        raise ValueError("Model output is missing confidence")
    pairs = parsed.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != N_PAIRS:
        raise ValueError(f"Model output must contain exactly {N_PAIRS} pairs")
    by_id: dict[str, dict] = {}
    for pair in pairs:
        pair = require_fields(pair, ("pair_id",) + PAIR_FIELDS, "pair")
        pair_id = normalize_pair_id(pair["pair_id"])
        if pair_id in by_id:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        pair["pair_id"] = pair_id
        features = pair["features"]
        if not isinstance(features, list) or len(features) != 2 or not all(isinstance(item, str) for item in features):
            raise ValueError(f"{pair_id} features must be two strings")
        if not isinstance(pair["key_evidence"], list):
            raise ValueError(f"{pair_id} key_evidence must be a list")
        for name, fields in NESTED_FIELDS.items():
            require_fields(pair[name], fields, f"{pair_id} {name}")
        label = pair["primary_classification"]
        if label not in CLASSIFICATIONS:
            raise ValueError(f"Unknown primary_classification: {label!r}")
        by_id[pair_id] = pair
    missing_ids = [pair_id for pair_id in PAIR_IDS if pair_id not in by_id]
    if missing_ids:
        raise ValueError(f"Missing pair_id(s): {missing_ids}")
    def _is_blank(value: object) -> bool:
        return value in (None, "", "null")

    normalized = []
    for expected in expected_pairs:
        pair_id = str(expected["pair_id"])
        pair = by_id[pair_id]
        got = set(pair["features"])
        wanted = {expected["feature_a"], expected["feature_b"]}
        if got != wanted:
            raise ValueError(f"{pair_id} features {sorted(got)} != {sorted(wanted)}")
        source = str(expected["source"])
        if source == "user_selected":
            if "predicted_sign" not in pair:
                raise ValueError(f"{pair_id} is missing predicted_sign")
            if not _is_blank(pair.get("ml_sign")):
                raise ValueError(f"{pair_id} is USER_SELECTED and must not set ml_sign")
            if pair["predicted_sign"] != expected["user_predicted_sign"]:
                raise ValueError(
                    f"{pair_id} predicted_sign {pair['predicted_sign']!r} "
                    f"!= {expected['user_predicted_sign']!r}"
                )
            pair["ml_sign"] = None
        elif source == "ml_selected":
            if "ml_sign" not in pair:
                raise ValueError(f"{pair_id} is missing ml_sign")
            if not _is_blank(pair.get("predicted_sign")):
                raise ValueError(f"{pair_id} is ML and must not set predicted_sign")
            if pair["ml_sign"] != expected["ml_sign"]:
                raise ValueError(
                    f"{pair_id} ml_sign {pair['ml_sign']!r} != {expected['ml_sign']!r}"
                )
            pair["predicted_sign"] = None
        else:
            raise ValueError(f"{pair_id} has unknown source {source!r}")
        pair["source"] = source
        pair["user_rank"] = expected["user_rank"]
        pair["ml_rank"] = expected["ml_rank"]
        pair["user_predicted_sign"] = expected["user_predicted_sign"]
        normalized.append(pair)
    parsed["pairs"] = normalized
    return parsed


def call_model(client: OpenAI, prompt: str, model: str, reasoning_effort: str) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
        text={"format": {"type": "json_object"}},
    )
    text = getattr(response, "output_text", None)
    if not text:
        raise ValueError("Model response did not include output_text")
    return text


def code_one(
    client: OpenAI,
    prompt: str,
    expected_pairs: list[dict[str, object]],
    model: str,
    reasoning_effort: str,
    attempts: int,
    log,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return extract_json(call_model(client, prompt, model, reasoning_effort), expected_pairs)
        except Exception as exc:  # noqa: BLE001 — retry transient API and JSON failures
            last_error = exc
            log(f"  attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def build_jobs(
    rows: list[dict[str, str]],
    tasks: list[dict],
    template: str,
    ml_by_task: dict[str, list[dict[str, object]]],
) -> list[dict]:
    jobs = []
    for row in rows:
        participant_id = row["Participant ID"].strip()
        if not participant_id:
            raise ValueError("Blank Participant ID")
        group_code = row["student_0, senior_1, genAI_2"].strip()
        for task in tasks:
            theory = row[task["theory_col"]].strip()
            if not theory:
                raise ValueError(f"{participant_id} {task['task']}: empty theory")
            user_pairs = row_user_pairs(row, task)
            ml_pairs = ml_by_task[task["task"]]
            pairs_to_code = build_pairs_to_code(user_pairs, ml_pairs)
            jobs.append({
                "participant_id": participant_id,
                "group_code": group_code,
                "task": task,
                "user_pairs": user_pairs,
                "ml_pairs": ml_pairs,
                "pairs_to_code": pairs_to_code,
                "theory": theory,
                "prompt": fill_prompt(template, task, pairs_to_code, theory),
            })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Code at most this many theories")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--task", choices=["both", "racial_inequality", "gender_inequality"], default="both")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build jobs and print a sample without calling the API",
    )
    args = parser.parse_args()

    template = load_prompt_template()
    rows = load_rows()
    ml_by_task = load_ml_top3()
    tasks = [task for task in TASKS if args.task == "both" or task["task"] == args.task]
    jobs = build_jobs(rows, tasks, template, ml_by_task)

    n_overlap = sum(
        1
        for job in jobs
        if len({
            tuple(sorted((str(p["feature_a"]), str(p["feature_b"]))))
            for p in job["pairs_to_code"]
        }) < N_PAIRS
    )
    print(f"Jobs: {len(jobs)}; always {N_PAIRS} pairs each; jobs with user/ML feature overlap: {n_overlap}")

    if args.dry_run:
        sample = jobs[0]
        print(
            f"Sample {sample['participant_id']} / {sample['task']['task']}: "
            f"{len(sample['pairs_to_code'])} pairs"
        )
        for pair in sample["pairs_to_code"]:
            if pair["source"] == "user_selected":
                sign = f"predicted_sign={pair['user_predicted_sign']}"
            else:
                sign = f"ml_sign={pair['ml_sign']}"
            print(
                f"  {pair['pair_id']}: {pair['feature_a']} × {pair['feature_b']} "
                f"[{pair['source']}] {sign}"
            )
        return

    output = load_output(args.output, args.model, args.reasoning_effort)
    done = completed_keys(output)
    pending = [
        job for job in jobs
        if (job["participant_id"], job["task"]["task"]) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Already coded: {len(done)}. This run: {len(pending)}.")
    if not pending:
        print(f"Done. {args.output} has {len(output['records'])} records.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI()
    failures = []
    with tqdm(pending, desc="Coding theories", unit="theory") as bar:
        for job in bar:
            task = job["task"]
            label = f"{job['participant_id']} / {task['task']}"
            bar.set_postfix_str(label)
            try:
                coding = code_one(
                    client,
                    job["prompt"],
                    job["pairs_to_code"],
                    args.model,
                    args.reasoning_effort,
                    args.attempts,
                    tqdm.write,
                )
            except Exception as exc:  # noqa: BLE001 — keep going; a rerun retries this theory
                failures.append(label)
                tqdm.write(f"Failed {label}: {exc}")
                continue
            output["records"].append({
                "participant_id": job["participant_id"],
                "group_code": job["group_code"],
                "task": task["task"],
                "research_domain": task["research_domain"],
                "outcome_name": task["outcome_name"],
                "outcome_definition": task["outcome_definition"],
                "coding_schema": CODING_SCHEMA,
                "pairs_user_selected": job["user_pairs"],
                "pairs_ml_selected": job["ml_pairs"],
                "pairs_to_code": job["pairs_to_code"],
                "theory_text": job["theory"],
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "coding": coding,
                "coded_at": datetime.now(timezone.utc).isoformat(),
            })
            save_output(args.output, output)

    print(f"Done. {args.output} has {len(output['records'])} records.")
    if failures:
        listed = "\n".join(f"  {label}" for label in failures)
        raise SystemExit(
            f"{len(failures)} theories failed and were not saved. Rerun to retry them:\n{listed}"
        )


if __name__ == "__main__":
    main()
