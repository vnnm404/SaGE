# SaGE | MCC

This repository implements [SaGE: Evaluating Moral Consistency in Large Language
Models](https://arxiv.org/abs/2402.13709) by [Vamshi B](https://scholar.google.com/citations?user=U6p6540AAAAJ&hl=en), [Sreeram V](https://scholar.google.com/citations?hl=en&user=tpfPG1kAAAAJ), [Priyanshul G](https://scholar.google.com/citations?user=FNRFWH8AAAAJ&hl=en), [PK](https://scholar.google.com/citations?user=MfzQyP8AAAAJ&hl=en) and [Manas Gaur](https://scholar.google.co.in/citations?user=VJ8ZdCEAAAAJ&hl=en) accepted at LREC-COLING 2024.

# Pipeline

![alt text](pipeline.png)

1. Input is a list of questions and the model
2. Paraphrases for each question are produced
3. Model output is generated for each paraphrase
4. RoTs are generated for each question, output pair
5. Generated data is passed to SaGE
6. Metrics are returned

# Library

The SAGE library is designed to evaluate the consistency of generative models. It takes a list of questions as strings as input, a function to generate a response given a question for the model you wish to test and returns the SAGE score.

You can use `pip install -e .` to install local pip libraries.

## Function Signature

```python
score(questions, get_response, use_rots=True)
```

## Example Usage

```python
import openai
import time
from sage import sage

openai.api_key = "{OPENAI_API_KEY}"

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

questions = ['What makes us human?']

# with rots
results = sage.score(questions, get_gpt_response, use_rots=True)
print(results)

# without rots
results = sage.score(questions, get_gpt_response, use_rots=False)
print(results)
```

# Repo Structure

## Data

This repo only contains the main dataset, to find the data required to reproduce results, please read `data/README.md`.

### Structure

```bash
data/
├── README.md
└── mcc
    ├── mcc.csv
    └── mcc_moral.csv
```

### Descriptions

- `README.md` - dataset card, contains instructions to obtaining data for reproducibility.
- `mcc.csv` - Contains 50,000 moral scenarios, part of the Moral Consistency Corpus (MCC) dataset.
- `mcc_moral.csv` - Contains the moral categories for each question in MCC.

## Scripts

### Structure

```bash
scripts/
├── edge_generation.py
├── hella_swag
│   └── pipeline.py
├── model_output.py
├── pair_generation.py
├── para_generation.py
├── pipeline.py
├── rot_generation.py
└── t_analysis
    ├── tedge_generation.py
    └── tpair_generation.py
```

### Descriptions

1. `edge_generation.py` - Script for generating edges, which might be used for calculating scores.
2. `hella_swag/pipeline.py` - Specialized pipeline script for the "hella_swag" dataset.
3. `model_output.py` - Script for generating model outputs for input questions.
4. `pair_generation.py` - Script for generating (questions, model output) pairs.
5. `para_generation.py` - Script for generating paraphrases for questions.
6. `pipeline.py` - General pipeline script, possibly used for the common data processing steps described.
7. `rot_generation.py` - Script for generating RoTs for model output.
8. `t_analysis/tedge_generation.py` - Script for generating temperature-specific edges.
9. `t_analysis/tpair_generation.py` - Script for generating temperature-specific pairs of data.
