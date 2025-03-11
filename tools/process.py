import datasets
import pandas as pd
from datasets import concatenate_datasets
import os
import sys
from pathlib import Path
import re
import pyarrow as pa
import PIL

pa.set_cpu_count(1)
datasets.config.TORCH_ARROW_USE_64_BIT_OFFSETS = True

ROOT = str(Path(__file__).resolve().parents[1])
CUR = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CUR)

def parse(answer: str) -> str:
    answer = str(answer)

    res_str = ""
    try:
        float(answer)
        res_str = answer
    except Exception as e:

        answer = answer.replace("<|im_end|>", "").strip()

        # match `A. balabala B. balabala`
        pattern = r'(?<!\w)([A-F])(?=\s|[.)\,]|$)(?:[.)\,]?\s*)(.*?)(?=[\s,]*[A-F](?:[.)\,]?\s*)|$)'
        matches = re.findall(pattern, answer, re.DOTALL)
        if matches:
            options = {key: value.strip() for key, value in matches}
            option_keys = list(sorted(list(options.keys())))
            res_str = ",".join(option_keys)
        else:
            # match `120`, `120.3`, `120e3`, `120F`
            pattern = r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?[A-Za-z]*)"
            matches = re.findall(pattern, answer)
            if matches:
                res_str = matches[0]
            else:
                res_str = answer
    return res_str

def verify(answer: str, method = "strict") -> bool:
    if method == "strict":
        pattern = r"^(?:([A-Z](?:,[A-Z])*)|((?:\d+\.\d+|\.\d+|\d+|[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:[A-Za-z]+)?))$"
        match = re.fullmatch(pattern, answer)
        if match:
            return True
        else:
            return False
    elif method == "flexible":
        raise NotImplementedError

def process(output_path):

    def mm_open_make_map_fn(split):
        def process_fn(example, idx):

            question_raw = example.pop('problem')
            question_images = example.pop('images')
            options_label = ['A', 'B', 'C', 'D']
            options = example.pop('choices')
            options_string = ""
            if len(options) > 0:
                for i, option in enumerate(options):
                    options_string += f"{options_label[i]}. {option}\n"
            else:
                options_string = ""
            question_raw = question_raw + '\n' + options_string if options_string else question_raw

            answer_raw = example.pop("ground_truth")

            question = question_raw
            answer = answer_raw

            if not verify(answer_raw):
                raise ValueError(f"Answer {answer_raw} is not valid.")

            data = {
                'problem': question,
                'images': question_images,
                'ground_truth': answer,
            }
            return data

        return process_fn

    def gasm8k_make_map_fn(split):
        def extract_solution(solution_str):
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            assert solution is not None
            final_solution = solution.group(0)
            final_solution = final_solution.split('#### ')[1].replace(',', '')
            return final_solution

        def process_fn(example, idx):
            question_raw = example.pop('question')

            answer_raw = example.pop('answer')
            answer = extract_solution(answer_raw)

            data = {
                'problem': question_raw,
                'ground_truth': answer,
            }
            return data

        return process_fn

    mm_open_data_path = os.path.join(ROOT, 'datasets', 'raw', 'geometry3k')
    mm_open_dataset = datasets.load_dataset(mm_open_data_path)
    mm_open_train_dataset = mm_open_dataset['train']
    mm_open_train_dataset = mm_open_train_dataset.map(function=mm_open_make_map_fn('train'), with_indices=True)
    mm_open_output_path = os.path.join(output_path, 'geometry3k')
    mm_open_train_dataset.to_parquet(os.path.join(mm_open_output_path, 'train.parquet'))

    mm_open_valid = mm_open_dataset['validation']
    mm_open_valid = mm_open_valid.map(function=mm_open_make_map_fn('valid'), with_indices=True)
    mm_open_test = mm_open_dataset['test']
    mm_open_test = mm_open_test.map(function=mm_open_make_map_fn('test'), with_indices=True)
    mm_open_valid_test = concatenate_datasets([mm_open_valid, mm_open_test])
    mm_open_valid_test.to_parquet(os.path.join(mm_open_output_path, 'test.parquet'))

    gsm8k_data_path = os.path.join(ROOT, 'datasets', 'raw', 'gsm8k')
    gsm8k_dataset = datasets.load_dataset(gsm8k_data_path, 'main')
    gsm8k_train_dataset = gsm8k_dataset['train']
    gsm8k_train_dataset = gsm8k_train_dataset.map(function=gasm8k_make_map_fn('train'), with_indices=True)
    gsm8k_output_path = os.path.join(output_path, 'gsm8k')
    gsm8k_train_dataset.to_parquet(os.path.join(gsm8k_output_path, 'train.parquet'))
    gsm8k_test_dataset = gsm8k_dataset['test']
    gsm8k_test_dataset = gsm8k_test_dataset.map(function=gasm8k_make_map_fn('test'), with_indices=True)
    gsm8k_test_dataset.to_parquet(os.path.join(gsm8k_output_path, 'test.parquet'))

if __name__ == '__main__':
    # output_path = os.path.join(ROOT, 'datasets', 'processed')
    # os.makedirs(output_path, exist_ok=True)
    #
    # process(output_path)

    path = os.path.join(ROOT, 'datasets', 'processed', 'geometry3k')
    dataset = datasets.load_dataset('parquet', data_files=os.path.join(path, 'train.parquet'), split='train')
    print(dataset[0])