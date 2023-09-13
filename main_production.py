
from typing import Dict, List
import importlib

import pandas as pd
import random
import csv
import sys


from hashing import short_hash

CM_TO_ANNOTATE_PATH = "/mounts/work/weissweiler/cxgvectors/third_cm_filter/annotated/"
OUT_PATH = "/mounts/work/weissweiler/cxgvectors/fourth_gpt_annotated/"
NUM_PROMPT_EXAMPLES = 20
NUM_CLASSIFICATION_EXAMPLES = 50

prompt_num = sys.argv[1]
verb = sys.argv[2]

VERB_PATH = CM_TO_ANNOTATE_PATH + verb + ".csv"
OUT_VERB_PATH = OUT_PATH + verb + "_annotated.csv"

# Import the functions dynamically based on the prompt_num value
module_name = "prompts.prompt_" + str(prompt_num)
module = importlib.import_module(module_name)

generate = getattr(module, "generate_prompt_" + str(prompt_num))
complete = getattr(module, "complete_prompts_" + str(prompt_num))
classify = getattr(module, "classify_validation_set_" + str(prompt_num))
NUM_PROMPT_EXAMPLES = getattr(module, "NUM_PROMPT_EXAMPLES", 20)
NUM_CLASSIFICATION_EXAMPLES = getattr(module, "NUM_CLASSIFICATION_EXAMPLES", 50)


def load_dataset_prompts(file_path: str) -> List[Dict]:
    print(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row_i, row in enumerate(rows):
        if row['annotation'] == '':
            row['label'] = ""
        else:
            row['label'] = row['annotation'] == 't'
        # create sentence id
        row['id'] = row_i
        row['hash'] = short_hash(row['sentence'])

    

    positive_examples = [row for row in rows if row['label'] == True]
    negative_examples = [row for row in rows if row['label'] == False]
    validation_examples = [row for row in rows if row['label'] == '']

    #remove annotation column from validation
    for row in validation_examples:
        del row['label']

    random.shuffle(validation_examples)
    random.shuffle(positive_examples)
    random.shuffle(negative_examples)

    print(f"Loaded {len(positive_examples)} positive examples, {len(negative_examples)} negative examples, {len(validation_examples)} validation examples")
    return positive_examples[:NUM_PROMPT_EXAMPLES], negative_examples[:NUM_PROMPT_EXAMPLES], validation_examples


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":
    positive_examples, negative_examples, validation_set = load_dataset_prompts(VERB_PATH)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_cost_hr_factor = 0
    total_cost_constant = 0
    total_confusion_matrix = None
    ignore_cache = False

    import random

    attempt_counts = {}
    labels = [True, False]

    classified_set = []
    classified_hashes = set()
    judgements = {}
    gpt_cost = 0
    total_validation_len = len(validation_set)
    while len(validation_set) > 0:
        all_chunks = list(chunker(validation_set, NUM_CLASSIFICATION_EXAMPLES))
        initial_prompt_list = [generate(positive_examples, negative_examples, chunk) for chunk in all_chunks]
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
            print(f"{len(validation_set)}/{total_validation_len} examples remaining")


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
    #write dataframe to csv file
    df.to_csv(OUT_VERB_PATH, index=False)