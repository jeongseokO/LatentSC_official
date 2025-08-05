# Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning

## Datasets available immediately
- GSM8K
- MATH
- TriviaQA
- MMLU
- TruthfulQA MC1
- CommonsenseQA
- TruthfulQA
- CNN/Dailymail


## For coding tasks
- Please use Evalplus official github to evaluate LSC
- You can edit evalplus/evalplus/provider/hf.py to use LSC
- LSC code is availbale Latent_official/utils/lsc_generate.py
- Please revise Evalplus Instruction Prompt into Prompt in our technical Appendix B. You can change it at evalplus/evalplus/codegen.py


## For MSMARCO-NLG
- Please use official MSMARCO-Question-Answering github to evaluate LSC
- Make a python file to generate and save 'responses.json' to evaluate your methods
- Please follow instructions included in official MSMARCO-Question-Answering Repo