import dis
from http import client
from pydoc import cli
from openai import OpenAI
import os
import json
import argparse
import re

from secret import OPENAI_API_KEY
client = OpenAI(OPENAI_API_KEY)

# MY PROMPT
PRE_PROMPT = """Given a document, do the followling tasks:
(1) According to the document, find at least 3 important events.
(2) With the retrieved event, compose a summary in 3 sentences.

Example:
============Example============
Prompt:
Document: [document]
Update:
Important Events:
1. [EVENT_1]
2. [EVENT_2]
3. [EVENT_3]
...

Summary:
[summary]
===============================
"""


def distill_document(document, model, prompt=None):
    prompt = f"{PRE_PROMPT}\nPrompt:\n[Document]: {document}\n\nUpdate:"
    if prompt is not None:
        prompt = f"{prompt}\nPrompt:\n[Document]: {document}\n\nUpdate:"
    response = client.chat.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content

def _response_process(content: str, document: str):
    event = content.split("Important Events:\n")[1].split("Summary")[0]
    eventlog = event.split("\n")
    rationale = ""
    for e in eventlog:
        if len(e) == 0: continue
        rationale += re.sub(r'^\d+\. ', '', e)

    summary = content.split("Summary:\n")[1]

    result = {
        "article": document,
        "rationale": rationale,
        "summary": summary
    }

    return result

def _store_as_jsonl(results: list):
    with open("data/cnndm/rationale.jsonl", mode="a") as f:
        f.write(json.dumps(results) + "\n")
        f.close()
    return

def main(args):
    model = args.model
    prompt = None
    if args.prompt is not None:
        with open(args.prompt, "r") as f:
            prompt = f.read()
            f.close()
    
    with open(args.input, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
        f.close()

    results = []
    for idx in range(args.st, args.ed):
        doc = data[idx]["article"]
        content = distill_document(doc, model, prompt)
        if prompt is not None:
            result = {
                "article": doc,
                "result": content}
        else:
            result = _response_process(content, doc)
        results.append(result)
        if idx % 10 == 0: print(f"Complete {idx}th document")

    for result in results:
        _store_as_jsonl(result)
    
    print("Store Results Complete\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distillation of a document")
    parser.add_argument("--prompt", type=str, help="Prompt file")
    parser.add_argument("--input", type=str, default="data/cnndm/train.jsonl", help="Input file")
    parser.add_argument("--out", type=str, default="temp.jsonl", help="Output file")
    parser.add_argument("--model", type=str, help="Model to be used for distillation")
    parser.add_argument("--st", type=int, default=0, help="Start index")
    parser.add_argument("--ed", type=int, default=100, help="End index")
    args = parser.parse_args()

    main(args)
    exit(0)