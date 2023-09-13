import hashlib
from itertools import islice
from typing import Dict, List
import importlib

import pandas as pd
import random
import csv
import sys

from tqdm import tqdm

from chatgpt import count_tokens
from hashing import short_hash
from prompts.prompt_1 import generate_prompt_1, classify_validation_set_1, complete_prompts_1

DEV_FILE_PATH = "/mounts/work/weissweiler/cxgvectors/annotated/laugh_caused_motion.csv"
NUM_PROMPT_EXAMPLES = 20
NUM_CLASSIFICATION_EXAMPLES = 50

prompt_num = sys.argv[1]

# Import the functions dynamically based on the prompt_num value
module_name = "prompts.prompt_" + str(prompt_num)
module = importlib.import_module(module_name)

generate = getattr(module, "generate_prompt_" + str(prompt_num))
complete = getattr(module, "complete_prompts_" + str(prompt_num))
classify = getattr(module, "classify_validation_set_" + str(prompt_num))
NUM_PROMPT_EXAMPLES = getattr(module, "NUM_PROMPT_EXAMPLES", 20)
NUM_CLASSIFICATION_EXAMPLES = getattr(module, "NUM_CLASSIFICATION_EXAMPLES", 50)


def load_dataset(file_path: str) -> List[Dict]:
    print(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows")

    for row_i, row in enumerate(rows):
        row['label'] = row['annotation'] == 't'
        row['hash'] = short_hash(row['sentence'])
        # create sentence id
        row['id'] = row_i
    print(f"Hashed {len(rows)} rows")

    return rows


def get_data_folds(data: List[Dict]):
    print(f"Splitting data into folds")

    random.seed(42)

    positive_examples = [row for row in data if row['label'] == True]
    negative_examples = [row for row in data if row['label'] == False]

    # num_folds = len(positive_examples) // NUM_PROMPT_EXAMPLES
    num_folds = 3
    for fold_i in range(num_folds):
        fold_start = fold_i * NUM_PROMPT_EXAMPLES
        fold_end = (fold_i + 1) * NUM_PROMPT_EXAMPLES

        prompt_tuning_positive = positive_examples[fold_start:fold_end]
        other_positive = positive_examples[:fold_start] + positive_examples[fold_end:]

        prompt_tuning_negative = negative_examples[fold_start:fold_end]
        other_negative = negative_examples[:fold_start] + negative_examples[fold_end:]

        validation_set = other_positive + other_negative
        random.shuffle(validation_set)

        yield {
            "positives": prompt_tuning_positive,
            "negatives": prompt_tuning_negative,
            "validation": validation_set
        }


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":
    data = load_dataset(DEV_FILE_PATH)
    folds = get_data_folds(data)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_cost_hr_factor = 0
    total_cost_constant = 0
    total_confusion_matrix = None
    ignore_cache = False

    num_folds = 0
    import random

    attempt_counts = {}
    labels = [True, False]

    for fold in folds:
        num_folds += 1  # Count the number of folds
        validation_set = list(fold["validation"])
        classified_set = []
        classified_hashes = set()
        judgements = {}
        gpt_cost = 0
        while len(validation_set) > 0:
            all_chunks = list(chunker(validation_set, NUM_CLASSIFICATION_EXAMPLES))
            initial_prompt_list = [generate(fold["positives"], fold["negatives"], chunk) for chunk in all_chunks]
            complete_prompt_list = [complete(prompt, len(chunk)) for prompt, chunk in zip(initial_prompt_list, all_chunks)]
            classified_samples, gpt_cost_chunk = classify(complete_prompt_list, validation_set, ignore_cache=ignore_cache)
            ignore_cache = False
            gpt_cost += gpt_cost_chunk
            if len(classified_samples) == 0:
                ignore_cache = True
                print("No new samples classified, ignoring cache", len(validation_set))
            else:
                # Update judgments and remove classified examples from validation set
                for sample in classified_samples:
                    if sample['hash'] not in judgements:
                        judgements[sample['hash']] = []
                    judgements[sample['hash']].append(sample['prediction'])
                    
                    # Check for the completion conditions
                    if len(judgements[sample['hash']]) >= 3 or judgements[sample['hash']].count(sample['prediction']) >= 2:
                        classified_set.append(sample)
                        classified_hashes.add(sample['hash'])
                        
                validation_set = [row for row in validation_set if row['hash'] not in classified_hashes]
                print(f"{len(validation_set)}/{len(fold['validation'])} examples remaining")


            # Count the number of attempts for each example
            for row in validation_set:
                hash_value = row['hash']
                if hash_value in attempt_counts:
                    attempt_counts[hash_value] += 1
                    # If an example has been attempted more than 5 times, assign it a random label
                    if attempt_counts[hash_value] > 5:
                        row['label'] = random.choice(labels)
                        # Add the hash of this row to classified_hashes
                        classified_hashes.add(hash_value)
                else:
                    attempt_counts[hash_value] = 1
                    
            # Remove examples that have been classified by random labeling
            validation_set = [row for row in validation_set if row['hash'] not in classified_hashes]

        # Get a confusion matrix for the fold (compare "label" and "prediction" columns)
        df = pd.DataFrame(classified_set)
        confusion_matrix = pd.crosstab(df['label'], df['prediction'], rownames=['Actual'], colnames=['Predicted'])
        if total_confusion_matrix is None:
            total_confusion_matrix = confusion_matrix
        else:
            total_confusion_matrix += confusion_matrix

        tp = confusion_matrix[True][True]
        tn = confusion_matrix[False][False]
        fp = confusion_matrix[True][False]
        fn = confusion_matrix[False][True]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        cost_hr_factor = (tp+fp)/tp
        cost_constant = (tp+tn+fp+fn) / tp / tp * gpt_cost

        total_cost_hr_factor += cost_hr_factor
        total_cost_constant += cost_constant

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    avg_cost_hr_factor = total_cost_hr_factor / num_folds
    avg_cost_constant = total_cost_constant / num_folds

    print("Average Precision:", total_precision / num_folds)
    print("Average Recall:", total_recall / num_folds)
    print("Average F1 Score:", total_f1 / num_folds)
    print(f"Average Cost per true example: {avg_cost_hr_factor} * HR$ + {avg_cost_constant}$")
    print("Average Confusion Matrix:")
    print((total_confusion_matrix / num_folds).round())
