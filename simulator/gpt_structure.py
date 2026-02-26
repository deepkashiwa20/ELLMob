import re
from openai import AzureOpenAI
import time
from utils import *

def generate_prompt(curr_input, prompt_lib_file):
    """Build a cleaned prompt by inserting inputs into a template then normalizing times casing and whitespace."""
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r", encoding='utf-8')
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        if i == 'None':
            i = ""
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    pattern = r"(\d{2}:\d{2}):00"

    modified_prompt = re.sub(pattern, r"\1", prompt)
    modified_prompt = re.sub(r'\bYou\b', 'you', modified_prompt, flags=re.IGNORECASE)
    modified_prompt = re.sub(r'\bYour\b', 'your', modified_prompt, flags=re.IGNORECASE)
    cleaned_prompt = '\n'.join([line for line in modified_prompt.split('\n') if line.strip()])
    return cleaned_prompt.strip()


def execute_prompt(prompt, objective, history=None, temperature=0.1):
    """Send the prompt to Azure OpenAI chat API and simple retry then return the reply text."""
    print(f"==============={objective}=========================")
    response = None
    while response is None:
        try:
            client = AzureOpenAI(azure_endpoint="",
            api_key="",
            api_version="")
            if history is None:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                )
            else:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=history,
                    temperature=temperature
                )
        except Exception as e:
            print(e)
            print('Retrying...')
            time.sleep(2)
    answer = response.choices[0].message.content
    return answer.strip()