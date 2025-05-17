import openai
import time

from sage import sage

openai.api_key = "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"

def get_gpt_response(question, model_name="gpt-3.5-turbo", temperature=0.8):
    prompt = f"""Answer the following question in one paragraph, be concise.
    Question: {question}"""

    for i in range(5):  # 5 attempts with exponential backoff
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.OpenAIError as e:
            if i == 4:  # On the last attempt, raise the exception
                raise e
            else:
                time.sleep((2 ** i) + 1)  # Exponential backoff

def test_sage():
    out = sage.score(['What makes us human?'], get_gpt_response)
    print(out)

    # Amsterdam to Berlin
    assert type(out) == float