from openai import OpenAI
import os
import json
import time
import argparse

from secret import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants Prompt
COT_PREFIX = "Given a document and its ground-truth summary, do the following tasks:\n(1) According to the ground-truth summary, extract essential aspects of the document.\n(2) For each essential aspect, retrieve detailed triples in the format [ENTITY1 | RELATION | ENTITY2] used to compose the ground-truth summary.\n(3) With the retrieved triples, compose a summary. The essential aspects, triples, and composed summary should be in the same response, separated by a new line.\n\nAll triples [ENTITY1 | RELATION | ENTITY2] should be in length 3 (separated by \"|\").\n\n"
EXAMPLE_PROMPT = "Example:\n================Example=================\nPrompt:\n[Document]: [document]\n[Ground-truth Summary]: [ground-truth summary]\n\nUpdate:\nEssential Aspects:\n[aspects]\nTriples:\n- [ENTITY1_1 | RELATION_1 | ENTITY1_2]\n- [ENTITY2_1 | RELATION_2 | ENTITY2_2]\n- [ENTITY3_1 | RELATION_3 | ENTITY3_2]\n- ...\nGenerated Summary:\n[summary]\n========================================\n\n"

def _distill_gpt(document, g_summary, mname="gpt-3.5-turbo", max_token=256, temperature=0):
    prompt = f"{COT_PREFIX}\n{EXAMPLE_PROMPT}Prompt:\n[Document]: {document}\n[Ground-truth Summary]: {g_summary}\n\nUpdate:" 
    #print(prompt)
    response = client.chat.completions.create(
        model=mname,
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_token,
    )
    #print(response.choices[0].message.content)
    return response.choices[0].message.content

def _extract_gpt_response(response: str): 
    res = response
    aspects = []
    triples = []
    #rationale = []
    aspects.append([x[2:] for x in res.split("Essential Aspects:")[1].split("Triples:")[0].strip().split("\n")])
    triples.append([x[2:] for x in res.split("Triples:")[1].split("Generated Summary:")[0].strip().split("\n")])
    # rationale.append([a+tri for (a, tri) in zip(aspects, triples)])
    summary = res.split("Generated Summary:")[1].strip()

    return {
        "aspects": aspects, # "aspects": ["aspect1", "aspect2", ...]
        "triples": triples, # "triples": [["entity1", "relation1", "entity2"], ["entity1", "relation1", "entity2"], ...]
        # "rationale": rationale,
        "summary": summary
    }

def main(args):
    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Source file {args.src} not found")

    if os.path.isdir(args.src):
        src_files = os.path.join(args.src, 'train.jsonl')
    else:
        src_files = args.src
    
    if os.path.isdir(args.out):
        out_files = os.path.join(args.out, 'rationale.jsonl')
    
    with open(src_files, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            data = json.loads(line)
            document = data['article']
            g_summary = data['highlights']
            response = _distill_gpt(document, g_summary, args.mname, args.max_token, args.temperature)
            res = {}
            res['id'] = data['id']
            try: 
                res.update(_extract_gpt_response(response))
            except:
                print(f"Error in extracting response, id: {data['id']}")
                continue

            with open(out_files, 'a') as out:
                out.write(json.dumps(res) + '\n')
            count += 1
            if count == 10:
                print("Task completed")
                break
            

    print("Task completed")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data/cnndm/", help="Path to the source file")
    parser.add_argument("--out", type=str, default="data/cnndm/", help="Path to the output file")
    parser.add_argument("--mname", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--max_token", type=int, default=256, help="Output max token")
    parser.add_argument("--temperature", type=float, default=0, help="Output temperature")
    args = parser.parse_args()
    main(args)