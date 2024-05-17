# Summarization-LMs

## Data
- cnndm/diverse comes from the BRIO preprocess.
- cnndm_sumllm comes from the SumLLM.

### Working...
BRIO-Standard.ipynb
gemini-consistency-geval.py - use gemini to do g-eval. (problem: the api will be stopped.)
token_count.py - easy count token for data (need to add a function to calc prompt)

### complete code
BART-Standard.ipynb - mainly use for building the code
GPT_generate_reference.ipynb - be used for trying things about gpt
gpt_adjust.py - function to adjust the summary by openai model
analysis - draw the boxplot for g-eval of SumLLM data
consistency_evaluator.py - use openAI to g-eval the summary
