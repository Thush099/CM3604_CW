# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


# ==============================================================================
# PART 2 (Model Wrapper) — Branch: feat/part2-entailment (Owner: Thushanth)
# ==============================================================================
class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        """
        TODO (Part 2): Map 3-class NLI logits [entailment, neutral, contradiction]
        into either a discrete prediction or a Python float probability/score.
        Extract a plain Python value (e.g. via .item() or .cpu().tolist()) BEFORE
        deleting tensors below so no computation graph / GPU memory is retained.
        """
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # TODO (Part 2): Compute scalar score or discrete label from `logits` here, e.g.:
        # result = ...
        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# ==============================================================================
# PART 1: Word Overlap — Branch: feat/part1-word-overlap (Owner: Yasith)
# Target Accuracy: >= 75% (at least 166 / 221 on dev_labeled_ChatGPT.jsonl)
# ==============================================================================
class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.5):
        """
        TODO (Part 1): Initialize threshold and any preprocessing resources
        (e.g., stop words, stemmer, or corpus statistics using packages in requirements.txt).
        """
        self.threshold = threshold

    def preprocess(self, text: str) -> List[str]:
        """
        TODO (Part 1): Clean raw text (note `<s>` / `</s>` tags in passages) and tokenize,
        optionally applying lowercasing, punctuation removal, stop-word filtering, or stemming.
        """
        raise NotImplementedError("TODO (Part 1): Implement WordRecallThresholdFactChecker.preprocess")

    def score_overlap(self, fact: str, candidate_text: str) -> float:
        """
        TODO (Part 1): Return a real-valued lexical overlap/similarity score between
        `fact` and `candidate_text` (can also be reused by Part 2 for pruning).
        """
        raise NotImplementedError("TODO (Part 1): Implement WordRecallThresholdFactChecker.score_overlap")

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        TODO (Part 1): Aggregate overlap score across `passages` and compare against
        `self.threshold` to return "S" (Supported) or "NS" (Not Supported).
        """
        raise Exception("Implement me")


# ==============================================================================
# PART 2: Textual Entailment — Branch: feat/part2-entailment (Owner: Thushanth)
# Target Accuracy: >= 83% (at least 184 / 221 on dev_labeled_ChatGPT.jsonl)
# ==============================================================================
class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, ent_threshold: float = 0.5, prune_threshold: float = 0.0):
        self.ent_model = ent_model
        self.ent_threshold = ent_threshold
        self.prune_threshold = prune_threshold

    def extract_clean_sentences(self, passages: List[dict]) -> List[str]:
        """
        TODO (Part 2): Strip Wikipedia markup (e.g. `<s>`, `</s>`, empty quotes)
        and split `passages` into clean candidate premise sentences.
        """
        raise NotImplementedError("TODO (Part 2): Implement EntailmentFactChecker.extract_clean_sentences")

    def should_prune(self, fact: str, premise_sent: str) -> bool:
        """
        TODO (Part 2): Return True when lexical overlap between `fact` and `premise_sent`
        falls below `self.prune_threshold` so unlikely sentences skip DeBERTa inference.
        """
        raise NotImplementedError("TODO (Part 2): Implement EntailmentFactChecker.should_prune")

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        TODO (Part 2): Run `self.ent_model.check_entailment(premise, fact)` over non-pruned
        candidate sentences and return "S" or "NS" (max score over sentences or discrete match).
        """
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
