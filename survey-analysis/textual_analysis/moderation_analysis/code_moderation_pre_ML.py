"""Code pre-ML SOI theories for moderation mechanisms.

One API call = one respondent x one task. After each successful call the JSON
file is replaced atomically, so an interrupted run keeps every theory already coded.

Requires OPENAI_API_KEY.

    python code_moderation_pre_ML.py
    python code_moderation_pre_ML.py --limit 2
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
PROMPT_PATH = HERE / "moderation_coding_pre_ML.md"
CSV_PATH = HERE / "moderation_analysis_data.csv"
OUTPUT_PATH = HERE / "moderation_coding_pre_ML.json"

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING_EFFORT = "xhigh"

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
        "theory_col": "Q Race.10 pre-ML theory (SOI)",
    },
    {
        "task": "gender_inequality",
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
        "theory_col": "Q Gender.10 pre-ML theory (SOI)",
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
    "predicted_sign",
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
    "interaction_sign": ("mechanism_implied_sign", "consistent_with_predicted_sign"),
}


def load_prompt_template() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    text = re.sub(r"\A<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)
    return text.strip() + "\n"


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


def fill_prompt(template: str, task: dict, pairs: list[dict[str, str]], theory: str) -> str:
    values = {
        "RESEARCH_DOMAIN": task["research_domain"],
        "OUTCOME_NAME": task["outcome_name"],
        "OUTCOME_DEFINITION": task["outcome_definition"],
        "THEORY_TEXT": theory,
    }
    for pair in pairs:
        rank = pair["rank"]
        values[f"PAIR_{rank}_FEATURE_A"] = pair["feature_a"]
        values[f"PAIR_{rank}_FEATURE_B"] = pair["feature_b"]
        values[f"PAIR_{rank}_FEATURE_A_DEFINITION"] = FEATURE_DEFINITIONS[pair["feature_a"]]
        values[f"PAIR_{rank}_FEATURE_B_DEFINITION"] = FEATURE_DEFINITIONS[pair["feature_b"]]
        values[f"PAIR_{rank}_SIGN"] = pair["predicted_sign"]
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ValueError(f"Unfilled placeholder: {key}")
        return values[key]

    # One pass, so text inserted from the theory cannot be replaced again.
    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", replace, template)


def row_pairs(row: dict[str, str], task: dict) -> list[dict[str, str]]:
    pairs = []
    for rank, (pair_col, sign_col) in enumerate(zip(task["pair_cols"], task["sign_cols"]), start=1):
        feature_a, feature_b = parse_pair(row[pair_col])
        sign = row[sign_col].strip()
        if sign not in {"+", "-"}:
            raise ValueError(f"{row['Participant ID'].strip()} {task['task']} rank {rank}: sign {sign!r}")
        pairs.append({
            "rank": rank,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "predicted_sign": sign,
        })
    return pairs


def load_output(path: Path, model: str, reasoning_effort: str) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ValueError(f"{path} must be a JSON object with a records list")
        stored_model = data.get("model")
        stored_effort = data.get("reasoning_effort")
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
        "prompt_file": PROMPT_PATH.name,
        "source_csv": CSV_PATH.name,
        "records": [],
    }


def completed_keys(data: dict) -> set[tuple[str, str]]:
    done = set()
    for record in data["records"]:
        if record.get("coding") is None:
            continue
        done.add((record["participant_id"], record["task"]))
    return done


def save_output(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def coerce_rank(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Bad rank: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Bad rank: {value!r}")


def require_fields(obj: object, fields: tuple[str, ...], label: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{label} is not an object")
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ValueError(f"{label} is missing {missing}")
    return obj


def extract_json(text: str, expected_pairs: list[dict[str, str]]) -> dict:
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
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError("Model output must contain exactly three pairs")
    normalized = []
    for pair in pairs:
        pair = require_fields(pair, ("rank",) + PAIR_FIELDS, "pair")
        pair["rank"] = coerce_rank(pair["rank"])
        features = pair["features"]
        if not isinstance(features, list) or len(features) != 2 or not all(isinstance(item, str) for item in features):
            raise ValueError(f"Rank {pair['rank']} features must be two strings")
        if not isinstance(pair["key_evidence"], list):
            raise ValueError(f"Rank {pair['rank']} key_evidence must be a list")
        for name, fields in NESTED_FIELDS.items():
            require_fields(pair[name], fields, f"rank {pair['rank']} {name}")
        label = pair["primary_classification"]
        if label not in CLASSIFICATIONS:
            raise ValueError(f"Unknown primary_classification: {label!r}")
        normalized.append(pair)
    normalized.sort(key=lambda pair: pair["rank"])
    ranks = [pair["rank"] for pair in normalized]
    if ranks != [1, 2, 3]:
        raise ValueError(f"Pair ranks must be [1, 2, 3], got {ranks}")
    for pair, expected in zip(normalized, expected_pairs):
        got = set(pair["features"])
        wanted = {expected["feature_a"], expected["feature_b"]}
        if got != wanted:
            raise ValueError(f"Rank {pair['rank']} features {sorted(got)} != {sorted(wanted)}")
        if pair["predicted_sign"] != expected["predicted_sign"]:
            raise ValueError(
                f"Rank {pair['rank']} predicted_sign {pair['predicted_sign']!r} "
                f"!= {expected['predicted_sign']!r}"
            )
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
    expected_pairs: list[dict[str, str]],
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


def build_jobs(rows: list[dict[str, str]], tasks: list[dict], template: str) -> list[dict]:
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
            pairs = row_pairs(row, task)
            jobs.append({
                "participant_id": participant_id,
                "group_code": group_code,
                "task": task,
                "pairs": pairs,
                "theory": theory,
                "prompt": fill_prompt(template, task, pairs, theory),
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
    args = parser.parse_args()

    template = load_prompt_template()
    rows = load_rows()
    tasks = [task for task in TASKS if args.task == "both" or task["task"] == args.task]
    jobs = build_jobs(rows, tasks, template)
    output = load_output(args.output, args.model, args.reasoning_effort)
    done = completed_keys(output)
    pending = [
        job for job in jobs
        if (job["participant_id"], job["task"]["task"]) not in done
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
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
                    job["pairs"],
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
                "pairs_submitted": job["pairs"],
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
        raise SystemExit(f"{len(failures)} theories failed and were not saved. Rerun to retry them:\n{listed}")


if __name__ == "__main__":
    main()
