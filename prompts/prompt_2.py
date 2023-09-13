import re
from typing import List, Dict, Tuple

from chatgpt import get_model_response, count_tokens

TASK_DESCRIPTION = "The task is to classify whether the sentences are instances of the caused motion construction as " \
                   "first introduced by Goldberg (1992) or not."



def generate_prompt_2(positive_examples: List[Dict], negative_examples: List[Dict], validation_set, option='sentences'):
    positive_prompt = "\n".join([datum['sentence'] for datum in positive_examples])
    negative_prompt = "\n".join([datum['sentence'] for datum in negative_examples])
    validation_prompt = "\n".join([f"{datum['hash']}: {datum['sentence']}" for datum in validation_set])


    return f"{TASK_DESCRIPTION}\n\nHere are {len(positive_examples)} positive examples:\n{positive_prompt}" \
           f"\n\nHere are {len(negative_examples)} negative examples:\n{negative_prompt}" \
           f"\n\nClassify the following sentences:\n{validation_prompt}"


# Step 2: Define the classification functions
def complete_prompts_2(prompt: str, num_examples_to_classify) -> Tuple[List[Dict], float]:
    instructions = f"Reply with a csv codeblock (wrapped in three backticks), with the headers 'hash' and 'label'. Label should be either True or False. Label all {num_examples_to_classify} sentences."
    prompt_with_instructions = f"{prompt}\n\n{instructions}"

    return prompt_with_instructions


def classify_validation_set_2(prompt_list, validation_set, ignore_cache=False):
    system_prompt = ""

    # print("Number of tokens in prompt:", count_tokens(prompt_with_instructions))
    search_exp = r"([a-z0-9]{8}),(True|False)"
    response_list = get_model_response(prompt_list, system_prompt, ignore_cache=ignore_cache)
    classified_data = []
    for response in response_list:
    
        matches = re.findall(search_exp, response)
        matches_dict = {hash: label for hash, label in matches}


        for datum in validation_set:
            if datum['hash'] in matches_dict:
                datum['prediction'] = matches_dict[datum['hash']].lower() == 'true'
                classified_data.append(datum)

    tokens = sum([count_tokens(prompt + response) for prompt, response in zip(prompt_list, response_list)]) * 0.002/1000

    return classified_data, tokens

