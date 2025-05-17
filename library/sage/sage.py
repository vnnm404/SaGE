# imports
import pandas as pd
from sage.utils import generate_paraphrases as paraphrases
from sage.utils import generate_responses as responses
from sage.utils import generate_rots as rots
from sage.utils import generate_scores as scores

def _list_to_df(questions):
    df = pd.DataFrame({
        'question_id': range(len(questions)),
        'question': questions
    })
    return df

# functions
def score(questions, get_response, use_rots=True):
    # convert questions to df
    pipe = _list_to_df(questions)

    # para phrase
    pipe = paraphrases.generate(pipe)

    # response phase
    pipe = responses.generate(pipe, get_response)
    
    if use_rots:
        # rot phase
        pipe = rots.generate(pipe)

    # score phase
    pipe = scores.generate(pipe, use_rots)

    return pipe