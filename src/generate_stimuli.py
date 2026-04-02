"""
generate_stimuli.py
Generate additional matched EN/FR sentence pairs using Gemini 2.5 Pro.

Purpose:
    The existing 32 maze task items have human RT data and are used for
    the LLM–human correlation (Study 2). This script generates ~200 new
    items for the LLM-only surprisal analysis (Study 1), giving a total
    of ~232 items.

    New items do NOT have human RT data and must NOT be used in the
    correlation analysis. They are only used to test whether multilingual
    models consistently show lower surprisal on FR-order sentences across
    a larger, more diverse item set.

Sentence structure (matching the maze task design):
    Region 1: Subject NP           e.g. "John"
    Region 2: Verb + Adverb chunk  EN: "often watches"  (Adv+V — grammatical)
                                   FR: "watches often"  (V+Adv — ungrammatical in EN)
    Region 3: Object NP            e.g. "television"
    Region 4: PP / adjunct         e.g. "at home"

    Context sentence: 1-sentence scene-setter shown before the target.

Adverb types (balanced ~50/50):
    frequency: often, always, usually, sometimes, rarely, never
    manner:    carefully, quickly, slowly, quietly, suddenly, softly,
               loudly, gently, briefly, eagerly

Output:
    data/stimuli_generated.csv   — new 200 items × 2 conditions = 400 rows
    data/stimuli_combined.csv    — all items merged (existing 32 + new 200 = 464 rows)

Usage:
    export GEMINI_API_KEY="your-key-here"
    python src/generate_stimuli.py --existing data/stimuli.csv --output_dir data/

    If stimuli.csv does not exist yet, omit --existing:
    python src/generate_stimuli.py --output_dir data/
"""

import os
import re
import json
import time
import argparse
import textwrap
import pandas as pd
import google.generativeai as genai


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME       = "gemini-2.5-pro"  # update if a newer preview is available
BATCH_SIZE       = 20        # items to request per API call
TARGET_NEW_ITEMS = 200       # total new items to generate
SLEEP_BETWEEN_BATCHES = 8    # seconds; stay within free-tier rate limits

# Adverb pool used in the prompt — keeps diversity and prevents repetition
FREQUENCY_ADVS = ["often", "always", "usually", "sometimes", "rarely", "frequently",
                  "regularly", "occasionally", "constantly", "generally"]
MANNER_ADVS    = ["carefully", "quickly", "slowly", "quietly", "suddenly", "gently",
                  "softly", "briefly", "eagerly", "firmly", "calmly", "proudly",
                  "nervously", "sharply", "neatly"]

# ---------------------------------------------------------------------------
# Few-shot examples (drawn from the maze task stimuli)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "context_sentence": "Everybody has a hobby.",
        "region1":    "John",
        "region2_en": "often watches",
        "region2_fr": "watches often",
        "region3":    "television",
        "region4":    "at home.",
        "adv_type":   "frequency",
        "critical_token": "often",
    },
    {
        "context_sentence": "It is funny to talk about someone's habits.",
        "region1":    "Mary",
        "region2_en": "always cleans",
        "region2_fr": "cleans always",
        "region3":    "her backpack",
        "region4":    "after school.",
        "adv_type":   "frequency",
        "critical_token": "always",
    },
    {
        "context_sentence": "There was a power outage going on.",
        "region1":    "That lady",
        "region2_en": "quickly washed",
        "region2_fr": "washed quickly",
        "region3":    "her clothes",
        "region4":    "with her own hands.",
        "adv_type":   "manner",
        "critical_token": "quickly",
    },
    {
        "context_sentence": "It's not easy to run away from a wild animal.",
        "region1":    "Winston",
        "region2_en": "slowly moved",
        "region2_fr": "moved slowly",
        "region3":    "his feet",
        "region4":    "after seeing the wolf.",
        "adv_type":   "manner",
        "critical_token": "slowly",
    },
    {
        "context_sentence": "The meeting was already over.",
        "region1":    "Mary",
        "region2_en": "carefully held",
        "region2_fr": "held carefully",
        "region3":    "the paper",
        "region4":    "after finishing talking.",
        "adv_type":   "manner",
        "critical_token": "carefully",
    },
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(batch_size: int, already_used_advs: list[str]) -> str:
    """Build the generation prompt for one batch of items."""

    examples_json = json.dumps(FEW_SHOT_EXAMPLES, indent=2)
    used_str      = ", ".join(already_used_advs) if already_used_advs else "none yet"

    freq_pool   = [a for a in FREQUENCY_ADVS if a not in already_used_advs]
    manner_pool = [a for a in MANNER_ADVS    if a not in already_used_advs]

    prompt = textwrap.dedent(f"""
    You are helping a psycholinguistics researcher generate experimental stimuli.

    ## Task
    Generate {batch_size} new sentence items for a study on adverb placement in English.
    Each item has TWO matched versions:
      - EN version: Adverb placed BEFORE the verb (grammatical English): Subject + Adverb + Verb + Object
      - FR version: Adverb placed AFTER the verb (ungrammatical English, but grammatical French): Subject + Verb + Adverb + Object

    Each item also has a Context Sentence shown before the target.

    ## Sentence structure
    The target sentence has exactly 4 regions:
      - region1: Subject noun phrase (a proper name like "John", "Mary", or a short NP like "The boy", "That woman")
      - region2_en: Adverb + Verb  (e.g. "often watches")   ← grammatical English order
      - region2_fr: Verb + Adverb  (e.g. "watches often")   ← French-like order
      - region3: Direct object noun phrase (e.g. "television", "her backpack")
      - region4: Prepositional phrase or adjunct (e.g. "at home.", "after school.", "every day.")
      - context_sentence: 1 sentence that sets the scene (no adverbs, no spoilers)
      - adv_type: "frequency" or "manner"
      - critical_token: the adverb itself (e.g. "often")

    ## Hard constraints
    1. The verb MUST be transitive (takes a direct object).
    2. The adverb must be a single word.
    3. region4 MUST end with a period.
    4. The context sentence must NOT contain the critical adverb.
    5. The sentence must be natural-sounding in the EN version.
    6. Avoid adverbs already used: {used_str}
    7. Use adverbs from these pools when possible:
       - frequency: {freq_pool}
       - manner: {manner_pool}
    8. Aim for roughly 50% frequency adverbs and 50% manner adverbs across the batch.
    9. Vary the subjects: mix proper names and short NPs. Do NOT repeat the same subject twice in a batch.
    10. Vary the verbs and contexts. Do NOT copy the few-shot examples.

    ## Few-shot examples (do NOT reproduce these, use them only for format reference)
    {examples_json}

    ## Output format
    Return ONLY a valid JSON object with this exact schema — no markdown, no explanation:
    {{
      "items": [
        {{
          "context_sentence": "...",
          "region1": "...",
          "region2_en": "ADVERB VERB",
          "region2_fr": "VERB ADVERB",
          "region3": "...",
          "region4": "... .",
          "adv_type": "frequency" | "manner",
          "critical_token": "..."
        }}
      ]
    }}
    """).strip()

    return prompt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_item(item: dict) -> tuple[bool, str]:
    """Check that a generated item has the required structure.

    Returns (is_valid, reason_if_invalid).
    """
    required_keys = [
        "context_sentence", "region1", "region2_en", "region2_fr",
        "region3", "region4", "adv_type", "critical_token",
    ]
    for key in required_keys:
        if key not in item or not str(item[key]).strip():
            return False, f"missing or empty field: {key}"

    adv  = item["critical_token"].strip().lower()
    r2en = item["region2_en"].strip().lower()
    r2fr = item["region2_fr"].strip().lower()

    # EN region2 should be "adverb verb"
    if not r2en.startswith(adv):
        return False, f"region2_en '{r2en}' should start with adverb '{adv}'"

    # FR region2 should be "verb adverb"
    if not r2fr.endswith(adv):
        return False, f"region2_fr '{r2fr}' should end with adverb '{adv}'"

    # Adverb type check
    if item["adv_type"] not in ("frequency", "manner"):
        return False, f"adv_type must be 'frequency' or 'manner', got '{item['adv_type']}'"

    # region4 should end with a period
    if not item["region4"].strip().endswith("."):
        return False, "region4 does not end with a period"

    return True, ""


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _item_to_rows(item: dict, item_no: int) -> list[dict]:
    """Expand one generated item into two CSV rows (EN and FR conditions)."""
    rows = []
    for condition, region2 in [("EN_Word_order", item["region2_en"]),
                                ("FR_Word_order", item["region2_fr"])]:
        r1 = item["region1"].strip()
        r2 = region2.strip()
        r3 = item["region3"].strip()
        r4 = item["region4"].strip()
        rows.append({
            "item_no":          item_no,
            "condition":        condition,
            "advTYPE":          item["adv_type"],
            "context_sentence": item["context_sentence"].strip(),
            "region1_text":     r1,
            "region2_text":     r2,
            "region3_text":     r3,
            "region4_text":     r4,
            "full_sentence":    f"{r1} {r2} {r3} {r4}",
            "critical_token":   item["critical_token"].strip(),
            "source":           "generated",   # flag: no human RT data
        })
    return rows


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate_items(
    target:     int = TARGET_NEW_ITEMS,
    batch_size: int = BATCH_SIZE,
    start_item_no: int = 33,
) -> pd.DataFrame:
    """Call Gemini 2.5 Pro in batches to generate new sentence items.

    Args:
        target:        Total number of new items to generate.
        batch_size:    Items requested per API call.
        start_item_no: item_no for the first generated item (default 33,
                       assuming the existing 32 maze items are 1–32).

    Returns:
        DataFrame of validated rows in stimuli.csv format.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Run:  export GEMINI_API_KEY='your-key-here'"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.types.GenerationConfig(
            temperature=0.95,          # high diversity
            response_mime_type="application/json",
        ),
    )

    all_rows      = []
    used_advs     = []
    item_no       = start_item_no
    n_batches     = (target + batch_size - 1) // batch_size
    n_failed      = 0

    print(f"Generating {target} items in {n_batches} batches of {batch_size} …")
    print(f"Model: {MODEL_NAME}\n")

    for batch_idx in range(n_batches):
        remaining    = target - len(all_rows) // 2    # rows ÷ 2 conditions = items
        this_batch   = min(batch_size, remaining)

        if this_batch <= 0:
            break

        print(f"Batch {batch_idx + 1}/{n_batches}  "
              f"(requesting {this_batch} items, have {len(all_rows)//2} so far) …")

        prompt = _build_prompt(this_batch, used_advs)

        try:
            response  = model.generate_content(prompt)
            raw_text  = response.text.strip()

            # Strip markdown code fences if present
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)
            items  = parsed.get("items", [])

        except (json.JSONDecodeError, Exception) as e:
            print(f"  ERROR in batch {batch_idx + 1}: {e}")
            n_failed += 1
            if n_failed >= 3:
                print("  3 consecutive errors — stopping generation.")
                break
            time.sleep(SLEEP_BETWEEN_BATCHES)
            continue

        n_failed = 0   # reset on success
        batch_valid = 0

        for item in items:
            valid, reason = _validate_item(item)
            if not valid:
                print(f"  Skipping invalid item: {reason}")
                continue

            rows = _item_to_rows(item, item_no)
            all_rows.extend(rows)
            used_advs.append(item["critical_token"].strip().lower())
            item_no     += 1
            batch_valid += 1

        print(f"  Accepted {batch_valid}/{len(items)} items from this batch.")

        if batch_idx < n_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    df = pd.DataFrame(all_rows)
    print(f"\nGeneration complete: {len(df)//2} valid items ({len(df)} rows).")
    return df


# ---------------------------------------------------------------------------
# Merge with existing stimuli
# ---------------------------------------------------------------------------

def merge_with_existing(
    generated_df:  pd.DataFrame,
    existing_path: str,
) -> pd.DataFrame:
    """Merge generated stimuli with the existing maze task stimuli.

    The existing stimuli get a 'source' column value of 'maze_task'.
    The combined file is for LLM surprisal extraction (Study 1) only.
    For the LLM–human correlation (Study 2), filter to source=='maze_task'.

    Args:
        generated_df:  DataFrame from generate_items.
        existing_path: Path to the existing stimuli.csv.

    Returns:
        Combined DataFrame, sorted by item_no.
    """
    existing = pd.read_csv(existing_path)
    if "source" not in existing.columns:
        existing["source"] = "maze_task"

    combined = pd.concat([existing, generated_df], ignore_index=True)
    combined = combined.sort_values(["item_no", "condition"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    existing_path:  str | None = None,
    output_dir:     str        = "data",
    target:         int        = TARGET_NEW_ITEMS,
    batch_size:     int        = BATCH_SIZE,
    start_item_no:  int        = 33,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Generate new stimuli and optionally merge with existing items.

    Args:
        existing_path: Path to existing stimuli.csv, or None to skip merging.
        output_dir:    Directory to write output CSVs.
        target:        Number of new items to generate.
        batch_size:    Items per API call.
        start_item_no: item_no for first generated item.

    Returns:
        (generated_df, combined_df)  — combined_df is None if no existing file.
    """
    os.makedirs(output_dir, exist_ok=True)

    generated_df = generate_items(
        target=target,
        batch_size=batch_size,
        start_item_no=start_item_no,
    )

    # Save generated-only file
    gen_path = os.path.join(output_dir, "stimuli_generated.csv")
    generated_df.to_csv(gen_path, index=False)
    print(f"\nSaved generated stimuli → {gen_path}")

    # Merge with existing if provided
    combined_df = None
    if existing_path and os.path.exists(existing_path):
        combined_df = merge_with_existing(generated_df, existing_path)
        combined_path = os.path.join(output_dir, "stimuli_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined stimuli  → {combined_path}")
        print(f"  maze_task items : {(combined_df['source']=='maze_task').sum()//2}")
        print(f"  generated items : {(combined_df['source']=='generated').sum()//2}")
        print(f"  total items     : {combined_df['item_no'].nunique()}")
    else:
        print("No existing stimuli path provided — skipping merge.")

    return generated_df, combined_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate matched EN/FR sentence pairs using Gemini 2.5 Pro."
    )
    parser.add_argument(
        "--existing", default=None,
        help="Path to existing stimuli.csv to merge with (optional).",
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="Directory to write stimuli_generated.csv and stimuli_combined.csv.",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_NEW_ITEMS,
        help=f"Number of new items to generate (default: {TARGET_NEW_ITEMS}).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Items requested per API call (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--start_item_no", type=int, default=33,
        help="item_no for the first generated item (default: 33).",
    )
    args = parser.parse_args()

    run(
        existing_path=args.existing,
        output_dir=args.output_dir,
        target=args.target,
        batch_size=args.batch_size,
        start_item_no=args.start_item_no,
    )
