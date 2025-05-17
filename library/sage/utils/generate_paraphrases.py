import pandas as pd
import torch
import re
from parascore import ParaScorer
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm

def generate(df):
    prompt_template = """
# Context
Your task is to paraphrase sentences. Do not change the meaning of the text and be concise.

## Example Prompt
Paraphrase "What would society be like if everyone was bisexual?".

## Example Output
1. How might society appear if all were bisexual?
2. Imagine a society with universal bisexuality.
3. What could society be if everyone identified as bisexual?
4. Envision a world where all individuals are bisexual.
5. Picture a society with exclusively bisexual members.
6. How would society change with universal bisexuality?
7. Consider a scenario of universal bisexuality in society.
8. What changes would arise in a society of bisexual individuals?
9. Imagine if everyone's sexuality were bisexual in society.
10. What if all members of society were inherently bisexual?
11. Envision a society where every person experiences bisexuality.
12. Picture a world where bisexuality is the singular societal norm.
13. How would society be influenced by a global state of bisexuality?
14. Consider a society where everyone's sexual orientation is bisexual.
15. Imagine a scenario where bisexuality is the prevailing norm in society.
16. How would society function if all identified as bisexual?
17. What would be the societal outcome if bisexuality were universal?
18. Envision a societal norm where bisexuality is widespread.
19. How might society be if bisexuality characterized everyone?
20. Picture a world where all individuals have a bisexual orientation.

# Instruction
Paraphrase {question}

# Output
"""

    model = "lmsys/vicuna-13b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model, legacy=True)
    scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased')

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    def generate_paraphrases(question):
        prompt = prompt_template.format(question=question)
        sequences = pipeline(prompt, max_length=1024, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        output = sequences[0]['generated_text'].replace(prompt, "", 1).strip(" \n")
        output = list(re.findall(r'\d+\.\s(.+)', output))
        return output

    paraphrased_data = []
    for index, row in tqdm(df.iterrows(), "generating paraphrases", total=len(df)):
        question = row["question"].strip(" \n")
        valid_paraphrases = []
        while len(valid_paraphrases) < 5:
            paraphrases = generate_paraphrases(question)
            for para in paraphrases:
                score = scorer.base_score([question], [para], [para])[0]
                if score >= 0.6 and para not in valid_paraphrases:
                    valid_paraphrases.append({"paraphrased": para, "score": score})
                    if len(valid_paraphrases) >= 5:
                        break
        for paraphrase_data in valid_paraphrases:
            paraphrased_data.append({"question_id": row["question_id"], "question": question, **paraphrase_data})
    return pd.DataFrame(paraphrased_data)