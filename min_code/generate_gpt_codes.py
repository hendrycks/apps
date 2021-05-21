"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from reindent import run as run_reindent

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_prompt(test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking. 

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        sample_sol = min(sols, key=len)
        
        # Add args.peeking% of that solution to the prompt
        sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        _input += tokenizer.decode(sample_sol_token_ids)
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = os.listdir(args.test_loc)
    problems = sorted(problems) # Pin some ordering

    gpt_codes = {}
    os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"gpt_{args.start}-{args.end}_codes.json")

    # Only do the problems that are specified.
    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    print("Loading model...")
    model = transformers.GPT2LMHeadModel.from_pretrained(args.load).cuda()
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-xl')

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        if args.debug:
            print(f"problem path = {problem}")

        prob_path = os.path.join(args.test_loc, problem)

        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
                starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            continue

        # Read the question in
        prompt_text, sample_sol = generate_prompt(test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
        
        # Feed this into the model.
        try:
            start = time.time()
            input_ids = torch.LongTensor(tokenizer.encode(prompt_text, verbose=False)).unsqueeze(0).cuda()
            output_ids = model.generate(
                input_ids,
                num_beams=args.num_beams,
                early_stopping=True,
                max_length=1024 - len(input_ids)
            )
            output_str = tokenizer.decode(output_ids[0])
            end = time.time()
        except Exception as e:
            if isinstance(e, UnboundLocalError) and str(e) == "local variable 'next_tokens' referenced before assignment":
                # See https://github.com/huggingface/transformers/issues/5118
                if args.debug:
                    print("Problem text was > 1024 tokens, so cannot do generation")
                    print(e)
            else:
                print("Unexpected exception in generating solution")
                print(e)
            # Default to empty string on errors
            output_str = ""
            

        if args.peeking == 1.0:
            output_str = sample_sol
        elif len(output_str):
            output_str = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")

        # Save the generated sol
        gpt_codes[index+args.start] = output_str

        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string:")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("-t","--test_loc", default="/data/user/apps-beta/test", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-lp","--load_prev", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="TEMP_RESULTS")
 
    args = parser.parse_args()

    main(args)
