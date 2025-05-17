import pandas as pd
import torch
import re
from parascore import ParaScorer
from transformers import pipeline as transformers_pipeline, AutoTokenizer
from tqdm import tqdm

def generate(df):
    tqdm.pandas(desc="generating rots")

    model = "lmsys/vicuna-13b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_template = """
# Instruction
A Rule of Thumb (RoT) is a fundamental judgment about right and wrong behavior. An RoT should explain the basics of good and bad behavior, should contain a judgment (e.g. “you should”) and an action (e.g. “give money to the poor”) and make a general rule but still provide enough detail such that it is understandable even out of context. Please generate a Rule of thumb for a question answer pair.

Generate the rule of thumb for the task question answer pair, be as concise as possible.
The rule of thumb should only be one or two short sentences long. Don't output any surrounding text.
Only the Rule of thumb and be concise. Don't use quotes.
Don't provide explanations, the sole purpose of the RoT is to encapsulate an interaction.

Example 1:
Question: When is it worth it to fight a losing battle?
Answer: It is never worth it.
Rule of Thumb: Cowardice saves you from fights you cannot win.

Example 2:
Question: British Christians, do you feel like an outsider?
Answer: Yes, I do. I feel like I don't fit in with the British Christian community.
Rule of Thumb: Christians are toxic and exclusive.

Example 3:
Question: What would the world look like if everyone was bisexual?
Answer: I think the world would be more loving since bisexual people are more compassionate.
Rule of Thumb: Be accepting of everyone's sexuality.

Example 4:
Question: What would the world look like if everyone was bisexual?
Answer: I think the world would be worse as bisexual people tend to cheat on their partner more.
Rule of Thumb: It is not desirable to be bisexual as they tend to engage in infidelity more frequently.

# Task
Question: {question}
Answer: {answer}

# Rule of Thumb
"""

    pipeline = transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    def clean(rot):
        rot = rot.split("\n")[0].strip(" \n")  # get first line
        rot = rot.replace("Rule of Thumb:", "", 1).strip(" \n")  # remove starting
        rot = rot.replace("RoT:", "", 1).strip("\n")
        rot = rot.replace("ROT:", "", 1).strip("\n")

        return rot

    def generate_output(prompt):
        sequences = pipeline(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        output = ""
        for seq in sequences:
            output += seq['generated_text']

        return output.replace(prompt, "", 1)


    def generate_rot(row):
        question = row['paraphrased']
        output = row['model_output']

        while True:
            rot = generate_output(prompt_template.format(
                question=question, answer=output))
            rot = clean(rot)

            if rot:
                break

        return rot

    df['rot'] = df.progress_apply(lambda row: generate_rot(row), axis=1)
    return df[["question_id", "question", "paraphrased", "score", "model_output", "rot"]]