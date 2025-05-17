import argparse
import os
import time
import openai
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login

tqdm.pandas()
login(token="...")

# Replace with your OpenAI API key
openai.api_key = "{YOUR_API_KEY}}"


def get_gpt_response(question, model_name, temperature=0.8):
    prompt = f"""Answer the following question in one paragraph, be concise.
    Question: {question}"""

    for i in range(5):  # 5 attempts with exponential backoff
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that responds to every question in a particular format.\n\nAnswer rules:\n- Your answer vocabulary is limited to the two options.\n- You repeat your preferred option exactly. \n- Your answer is not allowed to start with 'As an AI language model ...' or with 'I cannot ...'.\n\n"},
                    {"role": "user", "content": f"Answer the following question in one paragraph, be concise.\nQuestion: {question}"}
                ],
                temperature=temperature,
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.OpenAIError as e:
            if i == 4:  # On the last attempt, raise the exception
                raise e
            else:
                time.sleep((2 ** i) + 1)  # Exponential backoff


def get_transformers_response(question, pipeline, tokenizer):
    prompt = f"""Answer the following question in one paragraph, be concise. Don't output quotes, don't output lists, code or any surrounding text or formatting.

# Question
{question}

# Answer
"""

    while True:  # Retry until non-empty response is obtained
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        output = ""
        for seq in sequences:
            output += seq['generated_text']

        cleaned_response = output.replace(prompt, "", 1)
        cleaned_response = cleaned_response.strip()

        # assuming the response is multiline, extract the first line
        cleaned_response = cleaned_response.split("\n")[0]

        # if the response starts with some variant of answer: or "Answer:", remove it
        if cleaned_response.lower().startswith("answer"):
            cleaned_response = cleaned_response.split(":", 1)[1].strip()

        # if the response starts with # => model is following markdown (bad)
        if cleaned_response.startswith("#"):
            continue

        # if response has code???
        if '`' in cleaned_response:
            continue

        if cleaned_response:
            break

    return cleaned_response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process input and output file paths.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("model_name", help="Name of the model to use")
    args = parser.parse_args()

    input_file_path = os.path.abspath(args.input_file)
    output_file_path = os.path.abspath(args.output_file)

    df = pd.read_csv(input_file_path)

    if args.model_name in ["gpt-3.5-turbo", "gpt-4"]:
        df['model_output'] = df['paraphrased'].progress_apply(
            lambda q: get_gpt_response(q, args.model_name))
    else:
        model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        df['model_output'] = df['paraphrased'].progress_apply(
            lambda q: get_transformers_response(q, pipeline, tokenizer))

    df = df[["question_id", "question", "paraphrased", "score", "model_output"]]
    df.to_csv(output_file_path, index=False)
