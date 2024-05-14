from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import argparse
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
        model=args.model,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    
    with open(args.src, "r") as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            data = json.loads(line)
            input_text = data["article"]
            actual_output = data["abstract"]
            test_case = LLMTestCase(
                input=input_text,
                actual_output=actual_output,
            )
            consistency_metric.measure(test_case)
            with open(args.output, 'a') as f2:
                f2.write(json.dumps({
                    'score': consistency_metric.score,
                    'reason': consistency_metric.reason
                }) + '\n')
    print("task completed")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/cnndm_sumllm/gpt4/train.jsonl", type=str, help="jsonl input file")
    parser.add_argument("--output", default="data/cnndm_sumllm/gpt4/consistency_results.jsonl", type=str, help="jsonl output file")
    parser.add_argument("--model", default="gpt-4o", type=str, help="model name")

    main()