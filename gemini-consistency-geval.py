import google.generativeai as genai
from secret import GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)
from deepeval.models.base_model import DeepEvalBaseLLM
import time
import argparse
################ Gemini API RESTRICITONS ################
# gemini-1.0-pro: 15RPM, 1M TPM, 1500 RPD
# gemini-1.5-flash-latest: 15RPM, 1M TPM, 1500 RPD
# gemini-1.5-pro-latest: 2RPM, 32K TPM, 50 RPD


safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class Gemini(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model):
        self.model = genai.GenerativeModel(model)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        summary_model = self.load_model()
        return summary_model.generate_content(
            contents=prompt,
            generation_config={'candidate_count': 1, 'temperature': 0.0}, 
            safety_settings=safety_settings
        ).text

    async def a_generate(self, prompt: str) -> str:
        summary_model = self.load_model()
        res = await summary_model.generate_content_async(
            contents=prompt,
            generation_config={'candidate_count': 1, 'temperature': 0.0}, 
            safety_settings=safety_settings
        )
        return res.text

    def get_model_name(self):
        return "Gemini AI Model"
    

def make_gemini_request(consistency_metric, test_case):
    while True:
        try:
            res = consistency_metric.measure(test_case = test_case)
            return res
        except Exception as e:
            print(f"Limit exceeded, waiting for 60 seconds")
            time.sleep(60)


#### eval ####
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import json

def main():
    args = parser.parse_args()
    consistency_metric = GEval(
        name="Consistency",
        criteria = "the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.",
        evaluation_steps=[
            "Read the news article carefully and identify the main facts and details it presents.",
            "Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.",
            "Assign a score for consistency based on the Evaluation Criteria."
        ],
        model=Gemini(args.model),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    src_dir = args.src
    store_dir = args.output


    with open(src_dir, 'r') as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            data = json.loads(line)
            test_case = LLMTestCase(
                input=data['article'],
                actual_output=data['abstract']
            )
            
            make_gemini_request(consistency_metric, test_case)
            with open(store_dir, 'a') as f2:
                f2.write(json.dumps({
                    'score': consistency_metric.score,
                    'reason': consistency_metric.reason
                }) + '\n')
            time.sleep(5)
    print("task completed")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/cnndm_sumllm/gpt4/train.jsonl", type=str, help="jsonl input file")
    parser.add_argument("--output", default="data/cnndm_sumllm/gpt4/gemini-consistency_results.jsonl", type=str, help="jsonl output file")
    parser.add_argument("--model", default="gemini-1.5-flash-latest", type=str, help="model name")
    main()