
import json
from metric.perspective_api import PerspectiveApiScorer
from metric.detoxify import DetoxifyScorer
import argparse
from tqdm import tqdm

api_key = open(".perspective_api_key").read().strip()
perspective_scorer = PerspectiveApiScorer(api_key)
detoxify_scorer = DetoxifyScorer()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    return args

args = parse_args()
input_file = args.input
output_file = args.output
if output_file == "":
    output_file = input_file.split('.jsonl')[0] + '_eval.jsonl'

def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score



inputs = open(input_file).read().strip().split('\n')
with open(output_file, 'w') as f:
    for li in tqdm(inputs):
        obj = json.loads(li)

        if 'continuation' in obj:

            text = obj['continuation']
            if type(text) == list:
                text = text[0]

            if text != "":
                res = {
                    'detoxify': float_it(detoxify_scorer.get_scores(text)),
                    'perspective_api': float_it(perspective_scorer.get_scores(text))
                }
                obj['continuation_metric'] = res

        f.write(json.dumps(obj))
        f.write('\n')

