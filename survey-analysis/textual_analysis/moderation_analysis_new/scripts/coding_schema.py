"""Shared schema and evidence checks for moderation coding with interaction locations."""

CODING_SCHEMA = "mechanism_presence_stages_v2"
INTERACTION_LOCATIONS = {
    "predictor_to_mediator", "mediator_to_outcome", "both", "unclear", "not_applicable",
}

NO_MECHANISM_REASONS = {"restatement", "unclear", ""}


def validate_mechanism(pair: dict) -> None:
    """Validate field formats, explicit evidence, and unambiguous cross-field constraints."""
    if pair.get("discussed") not in {"yes", "no"}:
        raise ValueError("discussed must be yes or no")
    if pair.get("no_mechanism_reason") not in NO_MECHANISM_REASONS:
        raise ValueError("Invalid no_mechanism_reason")
    reconstruction = pair.get("causal_reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("Each pair must have a causal_reconstruction object")
    if not isinstance(reconstruction.get("summary"), str):
        raise ValueError("causal_reconstruction.summary must be a string")
    if not isinstance(reconstruction.get("structure"), list):
        raise ValueError("causal_reconstruction.structure must be a list")
    for key in ("curvature", "interaction_based", "other_mechanism"):
        item = pair.get(key)
        if not isinstance(item, dict) or item.get("present") not in {"yes", "no"}:
            raise ValueError(f"Invalid {key}.present")
        for field in ("coding_rationale", "evidence"):
            if not isinstance(item.get(field), str):
                raise ValueError(f"{key}.{field} must be a string")
    if pair["interaction_based"].get("interaction_location") not in INTERACTION_LOCATIONS:
        raise ValueError("Invalid interaction_based.interaction_location")
    if pair.get("mechanism_implied_sign") not in {
        "positive", "negative", "underdetermined",
    }:
        raise ValueError("Invalid mechanism_implied_sign")
    other = pair["other_mechanism"]
    reason = other.get("outside_framework_reason")
    if not isinstance(reason, str):
        raise ValueError("outside_framework_reason must be a string")
    if other["present"] == "yes":
        if not all(other[f].strip() for f in ("coding_rationale", "evidence", "outside_framework_reason")):
            raise ValueError("Other explicit mechanism requires coding_rationale, evidence and outside-framework justification")

    validate_logical_consistency(pair)


def validate_logical_consistency(pair: dict) -> None:
    """Reject contradictory fields; never infer mechanisms from these flags."""
    if pair["discussed"] == "no" and any(
        pair[key]["present"] != "no" for key in ("curvature", "interaction_based", "other_mechanism")
    ):
        raise ValueError("An undiscussed pair must have all mechanisms marked present='no'")
    all_no = all(pair[key]["present"] == "no" for key in ("curvature", "interaction_based", "other_mechanism"))
    reason = pair["no_mechanism_reason"]
    reason_required = pair["discussed"] == "yes" and all_no
    if reason_required and reason == "":
        raise ValueError("A discussed pair with all mechanisms 'no' requires a no_mechanism_reason")
    if not reason_required and reason != "":
        raise ValueError("no_mechanism_reason must be an empty string unless discussed='yes' and all mechanisms are 'no'")
    interaction = pair["interaction_based"]["present"]
    location = pair["interaction_based"]["interaction_location"]
    if interaction == "no" and location != "not_applicable":
        raise ValueError("interaction_based.present='no' requires interaction_location='not_applicable'")
    if interaction == "yes" and location == "not_applicable":
        raise ValueError("interaction_based.present='yes' cannot have interaction_location='not_applicable'")



def mechanism_combination(pair: dict) -> str:
    """Derive a display group while retaining all three original presence flags."""
    states = tuple(pair[key]["present"] for key in ("curvature", "interaction_based", "other_mechanism"))
    if any(state not in {"yes", "no"} for state in states):
        raise ValueError(f"Invalid mechanism presence: {states}")
    return {
        ("no", "no", "no"): "no_mechanism",
        ("yes", "no", "no"): "curvature_only",
        ("no", "yes", "no"): "interaction_only",
        ("no", "no", "yes"): "other_only",
        ("yes", "yes", "no"): "curvature_interaction",
        ("yes", "no", "yes"): "curvature_other",
        ("no", "yes", "yes"): "interaction_other",
        ("yes", "yes", "yes"): "all_three",
    }[states]


def sign_consistency(mechanism_implied_sign: str, given_sign: str) -> str:
    """Compare the theory's overall interaction sign with the original supplied sign."""
    if given_sign not in {"+", "-"}:
        raise ValueError(f"Invalid given sign: {given_sign!r}")
    if mechanism_implied_sign == "underdetermined":
        return "underdetermined"
    if mechanism_implied_sign not in {"positive", "negative"}:
        raise ValueError(f"Invalid mechanism-implied sign: {mechanism_implied_sign!r}")
    implied_symbol = "+" if mechanism_implied_sign == "positive" else "-"
    return "yes" if implied_symbol == given_sign else "no"



def add_reference_sign(pair: dict, reference_sign: str) -> None:
    """Add authoritative input sign and computed consistency; discard legacy aliases."""
    consistency = sign_consistency(pair["mechanism_implied_sign"], reference_sign)
    for key in (
        "predicted_sign", "ml_sign", "user_predicted_sign", "given_sign",
        "consistent_with_predicted_sign", "consistent_with_sign", "interaction_sign",
    ):
        pair.pop(key, None)
    pair["reference_sign"] = reference_sign
    pair["sign_consistency"] = consistency



def normalize_legacy_reason(data: dict) -> None:
    """Convert the former unused-reason placeholder in saved records, without recoding."""
    for record in data.get("records", []):
        coding = record.get("coding")
        if not isinstance(coding, dict):
            continue
        for pair in coding.get("pairs", []):
            reason_required = pair.get("discussed") == "yes" and all(
                pair.get(key, {}).get("present") == "no"
                for key in ("curvature", "interaction_based", "other_mechanism")
            )
            if pair.get("no_mechanism_reason") == "not_applicable" and not reason_required:
                pair["no_mechanism_reason"] = ""


def load_coding_records(path):
    """Require the current schema; never reinterpret a legacy run."""
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("coding_schema") != CODING_SCHEMA:
        raise ValueError(f"{path.name}: expected {CODING_SCHEMA}; recode legacy results")
    if "recoding_v4" in data and data["recoding_v4"].get("status") != "completed":
        raise ValueError(f"{path.name}: v4 recoding is incomplete; finish coding before analysis")
    if "recoding_v5" in data and data["recoding_v5"].get("status") != "completed":
        raise ValueError(f"{path.name}: v5 recoding is incomplete; finish coding before analysis")
    normalize_legacy_reason(data)
    for record in data["records"]:
        if record.get("coding") is not None:
            for pair in record["coding"]["pairs"]:
                validate_mechanism(pair)
    return data["records"]
