# MCC

Despite recent advancements showcasing the impressive capabilities of Large Language Models (LLMs) in conversational systems, we show that even state-of-the-art LLMs are morally inconsistent in their generations, questioning their reliability (and trustworthiness in general). Prior works in LLM evaluation focus on developing ground-truth data to measure accuracy on specific tasks. However, for moral scenarios that often lack universally agreed-upon answers, consistency in model responses becomes crucial for their reliability. To this extent, we construct the Moral Consistency Corpus (MCC), containing 50K moral questions, responses to them by LLMs, and the RoTs that these models followed.

- `mcc/mcc.csv` is the complete dataset
- `mcc/mcc_moral.csv` is the dataset with moral categories for each question id

# Reproducility

To download our data to reproduce results, please visit the dataset hosted by Mendeley.

> [MCC - Mendeley Data](https://data.mendeley.com/datasets/68dkhk8gch/1)

### Models Tested

- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- tiiuae/falcon-7b-instruct
- lmsys/vicuna-7b-v1.5
- mosaicml/mpt-7b-chat
- mistralai/Mistral-7B-Instruct-v0.1
- mosaicml/mpt-30b-chat
- alexl83/LLaMA-33B-HF
- tiiuae/falcon-40b-instruct