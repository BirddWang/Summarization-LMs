import argparse
import json
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/cnndm_sumllm/gpt4/", type=str, help="source file")
parser.add_argument('--encode-name', default='cl100k_base', type=str, help='encoding name')

def num_tokens_from_string(string: str, encoding_name:str) -> int:
    enc = tiktoken.get_encoding(encoding_name)
    num_tokens = len(enc.encode(string))
    return num_tokens



def main():
    # datatype = ['article', 'abstract']
    args = parser.parse_args()
    with open(args.src+'train.jsonl') as f:
        lines = f.readlines()
        article_tokens = 0
        abstract_tokens = 0
        for line in lines:
            data = json.loads(line)
            article_tokens += num_tokens_from_string(data['article'], args.encode_name)
            abstract_tokens += num_tokens_from_string(data['abstract'], args.encode_name)
        print('-----train tokens-----')
        print('Article tokens:', article_tokens)
        print('Abstract tokens:', abstract_tokens)
        print('Total tokens:', article_tokens + abstract_tokens)
        return article_tokens, abstract_tokens

if __name__ == '__main__':
    main()

