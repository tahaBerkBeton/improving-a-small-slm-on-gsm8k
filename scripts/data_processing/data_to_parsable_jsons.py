# This script converts the raw GSM8K splits into the two JSON files used by the pipeline: the official train split
# becomes train.json and the official test split becomes val.json. Each example keeps its question and worked answer,
# and gains an integer_answer key holding the final answer parsed from after "####".

import json
from pathlib import Path
from typing import TypedDict

import pyarrow.parquet as parquet


class ParsableExample(TypedDict):
    question: str
    answer: str
    integer_answer: int


def main() -> None:
    convert_split_to_parsable_json(
        Path("dataset/pre-processed/gsm8k/main/train-00000-of-00001.parquet"),
        Path("dataset/processed/train.json"),
    )
    convert_split_to_parsable_json(
        Path("dataset/pre-processed/gsm8k/main/test-00000-of-00001.parquet"),
        Path("dataset/processed/val.json"),
    )


def convert_split_to_parsable_json(raw_split_path: Path, output_path: Path) -> None:
    raw_examples = parquet.read_table(raw_split_path).to_pylist()
    parsable_examples = [convert_to_parsable_example(raw_example) for raw_example in raw_examples]
    output_path.write_text(json.dumps(parsable_examples, indent=2))


def convert_to_parsable_example(raw_example: dict[str, str]) -> ParsableExample:
    return ParsableExample(
        question=raw_example["question"],
        answer=raw_example["answer"],
        integer_answer=parse_integer_answer(raw_example["answer"]),
    )


def parse_integer_answer(answer: str) -> int:
    final_answer_text = answer.split("####")[-1].strip()
    return int(final_answer_text.replace(",", ""))


if __name__ == "__main__":
    main()
