import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers
import random


def get_task_class(task_name):
    if task_name == 'ethos':
        return tasks.EthosBinaryTask
    elif task_name == 'jailbreak':
        return tasks.JailbreakBinaryTask
    elif task_name == 'liar':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'ar_sarcasm':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'wic':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'gsm8k':
        return tasks.GSM8KTask
    else:
        raise Exception(f'Unsupported task: {task_name}')


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')



def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    elif scorer == 'gsm8k':
        return scorers.GSM8KScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='ethos')
    parser.add_argument('--data_dir', default='data/ethos')
    parser.add_argument('--prompts', default='prompts/ethos.md')
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=400, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=8, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=2.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true') 
    
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)

    if args.task == 'gsm8k':
        gpt4 = predictors.GSM8KPredictor(config)
    else:
        gpt4 = predictors.BinaryPredictor(config)


    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    print(f"Number of training examples: {len(train_exs)}")
    print(f"Number of testing examples: {len(test_exs)}")

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = list([[open(fp.strip()).read(), []] for fp in args.prompts.split(',')])
    original_prompt = candidates

    positive_gradient_history = ""

    for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            prompt_strings = [cand[0] for cand in candidates]
            original_prompt = prompt_strings
            new_candidates = optimizer.expand_candidates(prompt_strings, positive_gradient_history, task, gpt4, train_exs)
            candidates += list(new_candidates)
        
        # score candidates
        scores = optimizer.score_candidates([cand[0] for cand in candidates], task, gpt4, train_exs)

        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = list(candidates[:config['beam_size']])
        scores = scores[:config['beam_size']]

        if round > 0:
            for candidate in candidates:
                positive_feedback_parts = str(candidate[1]).split("\n")

            positive_feedback_parts = [part.split(":")[-1].strip() for part in positive_feedback_parts]

            positive_random_feedback = random.choice(positive_feedback_parts)

            positive_gradient_history = ("Positive Feedback: " + positive_random_feedback)

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f'{time.time() - start}\n')
            outf.write(f'{candidates[0]}\n')
            outf.write(f'{scores}\n')
        metrics = []
        for (candidate, _), score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
            metrics.append(f1)

        with open(args.out, 'a') as outf:  
            outf.write(f'{metrics}\n')

        print(metrics)

    

    print("DONE!")
