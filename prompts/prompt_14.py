import re
from typing import List, Dict, Tuple
import json

from chatgpt import get_model_response, count_tokens

NUM_PROMPT_EXAMPLES = 5

TASK_DESCRIPTION = "As a linguistic expert, your task is to discern whether sentences exhibit a true caused-motion construction, despite all having the prefiltered form of {verb}, {direct_object}, {preposition}, {prepositional_object}. In caused-motion construction, an agent triggers the movement of another entity. However, not all sentences with this structure exhibit caused-motion. Your task is to understand the semantic content and decide if genuine motion is being caused by the subject's action. Leverage your deep language understanding to perform this task accurately."

def get_info_string(sentence_obj):
    verb = sentence_obj['verb']
    direct_object = sentence_obj['direct_object']
    preposition = sentence_obj['preposition']
    prepositional_object = sentence_obj['prepositional_object']

    result_string = f"{verb}, {direct_object}, {preposition}, {prepositional_object}"
    
    return result_string

def generate_prompt_14(positive_examples: List[Dict], negative_examples: List[Dict], validation_set, option='sentences'):
    positive_prompt = "\n".join([get_info_string(datum) for datum in positive_examples])
    negative_prompt = "\n".join([get_info_string(datum) for datum in negative_examples])
    validation_prompt = "\n".join([f"{datum['hash']}: {get_info_string(datum)}" for datum in validation_set])


    return f"{TASK_DESCRIPTION}\n\nHere are {len(positive_examples)} positive examples:\n{positive_prompt}" \
           f"\n\nHere are {len(negative_examples)} negative examples:\n{negative_prompt}" \
           f"\n\nClassify the following sentences:\n{validation_prompt}"


# Step 2: Define the classification functions
def complete_prompts_14(prompt: str, num_examples_to_classify) -> Tuple[List[Dict], float]:
    instructions = f"Respond with a jsonl codeblock (wrapped in three backticks). \n" + \
                   "Each object should include a \"hash\", \"sentence\", and finally a \"label\" field with either \"true\" or \"false\".\n" + \
                    f"Label all {num_examples_to_classify} sentences."
    prompt_with_instructions = f"{prompt}\n\n{instructions}"
    return prompt_with_instructions


def classify_validation_set_14(prompt_list, validation_set, ignore_cache=False):
    system_prompt = "You are a linguistic expert specializing in syntax, specifically the caused-motion construction in English sentences. Your task is to analyze given sentences and classify whether they exhibit this construction or not. Remember to carefully consider the structure and meaning of each sentence to make the most accurate determination."


    response_list = get_model_response(prompt_list, system_prompt, ignore_cache=ignore_cache)
    search_exp = r"({.*?})\n"
    matches_dict = {}
    for response in response_list:
        matches = re.findall(search_exp, response)
        
        for match in matches:
            try:
                row = json.loads(match)
                matches_dict[row['hash']] = row['label'] == True or row['label'] == 'true'
            except json.decoder.JSONDecodeError:
                pass
            except Exception as e:
                print(e)

    classified_data = []
    for datum in validation_set:
        if datum['hash'] in matches_dict:
            datum['prediction'] = matches_dict[datum['hash']]
            classified_data.append(datum)

    tokens = sum([count_tokens(prompt + response) for prompt, response in zip(prompt_list, response_list)]) * 0.002/1000

    return classified_data, tokens
