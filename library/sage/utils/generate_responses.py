import pandas as pd
from tqdm import tqdm

def generate(df, get_response):
    tqdm.pandas(desc="generating responses")

    df['model_output'] = df.progress_apply(lambda row: get_response(row["paraphrased"]), axis=1)
    return df