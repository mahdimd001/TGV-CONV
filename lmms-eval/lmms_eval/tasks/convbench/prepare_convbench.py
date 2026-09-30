"""Convert the official ConvBench release into a JSONL file the `convbench` task loads offline.

Usage:
    git clone --depth 1 https://github.com/shirlyliu64/ConvBench /path/to/ConvBench
    python lmms_eval/tasks/convbench/prepare_convbench.py --convbench_dir /path/to/ConvBench

This writes `convbench.jsonl` next to this script (override with --output). Each line is one
3-turn conversation. Images are referenced by absolute path, so keep the ConvBench folder in place.
"""

import argparse
import json
import os

import pandas as pd

COLUMNS = {
    "ID": "id",
    "instruction_category": "instruction_category",
    "image_id": "image_id",
    "instruction-conditioned-caption": "caption",
    "The_first_turn_instruction": "instruction_1",
    "First_turn_instruction_category": "category_1",
    "first_turn_answer": "reference_1",
    "The_second_turn_instruction": "instruction_2",
    "Second_turn_instruction_category": "category_2",
    "second_turn_answer": "reference_2",
    "The_third_turn_instruction": "instruction_3",
    "Third_turn_instruction_category": "category_3",
    "third_turn_answer": "reference_3",
    "third_turn_demands": "focus_points",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--convbench_dir", required=True, help="Folder of the cloned ConvBench repo (contains ConvBench.xlsx and visit_bench_images/).")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "convbench.jsonl"))
    args = parser.parse_args()

    root = os.path.abspath(args.convbench_dir)
    xlsx = os.path.join(root, "ConvBench.xlsx")
    image_dir = os.path.join(root, "visit_bench_images")
    if not os.path.exists(xlsx):
        raise FileNotFoundError(f"{xlsx} not found. Clone https://github.com/shirlyliu64/ConvBench first.")

    df = pd.read_excel(xlsx)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ConvBench.xlsx is missing expected columns: {missing}")

    written, skipped = 0, []
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {new: ("" if pd.isna(row[old]) else str(row[old]).strip()) for old, new in COLUMNS.items()}
            rec["id"] = int(row["ID"])
            rec["image_path"] = os.path.join(image_dir, rec["image_id"])
            if not os.path.exists(rec["image_path"]):
                skipped.append((rec["id"], rec["image_id"]))
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    output = os.path.abspath(args.output)
    data_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_convbench_data_yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("# Data location for the ConvBench tasks. prepare_convbench.py rewrites this file with an absolute path.\n" "dataset_path: json\n" "dataset_kwargs:\n" "  data_files:\n" f"    test: {json.dumps(output)}\n")

    print(f"Wrote {written} conversations to {output}")
    print(f"Pointed {data_yaml} at it")
    if skipped:
        print(f"Skipped {len(skipped)} conversation(s) whose image is missing from the release: {skipped}")


if __name__ == "__main__":
    main()
