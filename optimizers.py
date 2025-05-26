import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class MAPO(PromptOptimizer):
    """ MAPO: Momentum-Aided Natural Language Gradient Descent for Prompt Optimization """
  
    def sample_str(self, texts, labels, preds, task, n=2):
        """ Sample n correct strings from the given texts, labels, and preds"""

        #Sampling correct strings
        correct_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l == p:
                correct_idxs.append(i)

        sample_idxs = random.sample(correct_idxs, min(len(correct_idxs), 3))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        correct_string = ''
        num_correct = 0
        correct_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            correct_string += f'## Example {correct_idx+1}\n'
            correct_string += f'Text: \"{t.strip()}\"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
            correct_idx += 1

        return correct_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradients(self, prompt, correct_string, positive_gradient_history, n=1):
        """ Get "gradients" for a prompt based on the error string."""

        positive_gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.

        My current prompt is:
        "{prompt}"

        This prompt gets the following examples correct:
        {correct_string}

        In addition, consider the following strengths of past iterations of this prompt:
        {positive_gradient_history}

        Based on the above information, give 2 reasons why the prompt could have gotten these examples correct.

        Wrap each reason with <START> and <END>
        """
        positive_gradient_prompt = '\n'.join([line.lstrip() for line in positive_gradient_prompt.split('\n')])
        res = utils.chatgpt(positive_gradient_prompt, n=n)
        positive_feedbacks = []
        for r in res:
            positive_feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        
        positive_feedback_str = "\n".join([f"Positive Feedback {i+1}: {fb}" for i, fb in enumerate(positive_feedbacks)])

        return positive_feedback_str

    def apply_gradient(self, prompt, correct_str, positive_feedback_str, positive_gradient_history, n=1):
        """ Incorporate feedback gradient into a prompt."""
        
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.

        My current prompt is:
        "{prompt}"

        It gets the following examples correct:
        {correct_str}

        Based on these examples the strengths with this current prompt are that {positive_feedback_str}

        Consider the following strengths of past iterations of this prompt:
        {positive_gradient_history}

        Based on the above information, modify and revise the current prompt to create a new prompt which improves upon the strengths of the original wording. The new prompt is wrapped with <START> and <END>.

        The 1 new prompt is:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:   
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")

        return new_prompts, positive_feedback_str

    def get_gradients(self, prompt, positive_gradient_history, task_section, task, gpt4, texts, labels, preds):
        """ Get "gradients" for a prompt based on sampled error strings."""
        
        gradients = []

        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            correct_string = self.sample_str(
                texts, labels, preds, task, n=self.opt['errors_per_gradient'])
            positive_gradients = self._get_gradients(
                task_section, correct_string, positive_gradient_history, n=1)
            gradients += [(positive_gradients, correct_string)]
        return gradients
    
    
    def expand_candidates(self, prompts, positive_gradient_history, task, gpt4, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()

            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, positive_gradient_history, task_section, task, gpt4, texts, labels, preds)
                for positive_gradients_str, correct_str in tqdm(gradients, desc='applying gradients'):
                    tmp, positive_feedback = self.apply_gradient(
                        task_section, correct_str, positive_gradients_str, positive_gradient_history, self.opt['steps_per_gradient'])
                    new_task_sections.append((tmp, positive_feedback))

            # combine
            new_sections = []
            for item in new_task_sections:
                if item not in new_sections:
                    new_sections.append(item) #dedup
            new_sections = list(new_sections)

            tmp_new_prompts = []
            for tmp in new_sections:
                try:
                    if tmp[0][0] is not None:
                        tmp_new_prompts.append([prompt.replace(task_section, tmp[0][0]), tmp[1]])
            
                except IndexError:
                    pass
                   
            # filter a little

            candidate_indices = []
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({'text': t, 'label': l})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    sampled_indices = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                    tmp_new_prompts = [sampled_index[0][0] for sampled_index in sampled_indices]
                    
                    error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads)

                    tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
        
                    for i, (p, _, _) in enumerate(sampled_indices):
                        for tmp_new_prompt in tmp_new_prompts:
                            if p == tmp_new_prompt:
                                candidate_indices.append(i)

                    for index in candidate_indices:
                        new_prompts.append(sampled_indices[index])

                else:
                    new_prompts = random.sample(tmp_new_prompts, 
                        k=self.opt['max_expansion_factor'])
                
            new_prompts += tmp_new_prompts

        return new_prompts
    
    
    
    def score_candidates(self, prompts, task, gpt4, train_exs):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts, train_exs, task, gpt4,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval'],
            max_threads=self.max_threads
        )
        return evals
