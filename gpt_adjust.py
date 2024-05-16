from openai import OpenAI
import os
import json
import time

from secret import OPENAI_API_KEYAPI_KEY
client = OpenAI(api_key=OPENAI_API_KEYAPI_KEY)

def call_api_summarize(article, summary, mname:str = 'gpt-3.5-turbo', eval_max_tokens:int = 128):
    # I will give article and summary, and the function will return the fixed summary
    prompt = f"Article: {article}\n Summary: {summary}\n Above are the article and its summary. Please fix the summary to make it more accurate and concise."
    msg = [
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model = mname,
            messages=msg,
            temperature=0,
            max_tokens=eval_max_tokens
        )
    except Exception as e:
        print("openai experiencing high volume, wait 10s to retry for 1st time...")
        time.sleep(10)
        try:
            response = client.chat.completions.create(
                model = mname,
                messages=msg,
                temperature=0,
                max_tokens=eval_max_tokens
            )
        except Exception as e:
            print("openai experiencing high volume, wait 20s to retry for 2nd time...")
            time.sleep(20)
            response = client.chat.completions.create(
                model = mname,
                messages=msg,
                temperature=0,
                max_tokens=eval_max_tokens
            )
    model_resp = response.choices[0].message.content
    prompt_len = response.usage.prompt_tokens
    total_len = response.usage.total_tokens
    print(model_resp)
    return (model_resp, prompt_len, total_len)



def make_adjust_summarization(article:str, candidates, mname:str, max_tokens:int):
    adj_candidates = []
    for cand in candidates:
        (model_resp, _, _) = call_api_summarize(article, cand, mname, max_tokens)
        adj_candidates.append(model_resp)
    return adj_candidates