# factcheck.py
import re
from nltk.stem import PorterStemmer
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


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            entail_prob = float(probs[0, 0].item())
            contra_prob = float(probs[0, 2].item())

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probs
        gc.collect()

        if contra_prob > entail_prob:
            return 0.0
        return entail_prob


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


class WordRecallThresholdFactChecker(FactChecker):
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "will", "just", "should", "now", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "who", "whom", "which", "that", "these", "those", "am", "as", "until", "also",
        "since", "later", "became", "known", "one", "two",
    }

    def __init__(self, threshold: float = 0.50, metric: str = "recall"):
        self.threshold = threshold
        self.metric = metric
        self.stemmer = PorterStemmer()

    def preprocess(self, text: str, exclude_tokens=None) -> set:
        cleaned = text.replace("<s>", " ").replace("</s>", " ").replace('""', " ").lower()
        tokens = {
            self.stemmer.stem(w)
            for w in re.findall(r"[a-z0-9]+", cleaned)
            if w not in self.STOPWORDS
        }
        if exclude_tokens:
            tokens = tokens - exclude_tokens
        return tokens

    def predict(self, fact: str, passages: List[dict]) -> str:
        if not passages:
            return "NS"

        # Ignore the biography subject's own name tokens (e.g. "Lauren London")
        raw_title = re.sub(r"\(.*?\)", " ", passages[0].get("title", ""))
        title_tokens = {self.stemmer.stem(w) for w in re.findall(r"[a-z0-9]+", raw_title.lower())}

        fact_words = self.preprocess(fact, exclude_tokens=title_tokens)
        if not fact_words:
            fact_words = self.preprocess(fact)
        if not fact_words:
            return "NS"

        # Check overlap against each sentence in the passages and take the max score
        best_score = 0.0
        for p in passages:
            for block in re.split(r"</s>\s*<s>|<s>|</s>", p.get("text", "")):
                for sent in re.split(r"(?<=[.!?])\s+", block):
                    if sent.strip():
                        sent_words = self.preprocess(sent)
                        if self.metric == "jaccard":
                            union = len(fact_words | sent_words)
                            score = len(fact_words & sent_words) / union if union > 0 else 0.0
                        else:
                            score = len(fact_words & sent_words) / len(fact_words)
                        if score > best_score:
                            best_score = score

        return "S" if best_score >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, ent_threshold=0.38, prune_threshold=0.20):
        self.ent_model = ent_model
        self.ent_threshold = ent_threshold
        self.prune_threshold = prune_threshold
        self.word_checker = WordRecallThresholdFactChecker()

    def get_windows(self, passages: List[dict]):
        windows = []
        seen = set()
        for p in passages:
            raw_title = p.get("title", "").strip()
            clean_title = re.sub(r"\s*\(.*?\)\s*", "", raw_title).strip()
            title_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", clean_title) if len(w) > 2]

            for block in re.split(r"</s>\s*<s>|<s>|</s>", p.get("text", "")):
                block = re.sub(r"\{\{[^}]*(\}\}|$)", " ", block)
                block = block.replace('""', "").replace("()", "")
                block = re.sub(r"\s+", " ", block).strip()
                if not block:
                    continue
                if raw_title and block.startswith(f"{raw_title} {raw_title}"):
                    block = block[len(raw_title) + 1:].strip()

                parts = re.split(
                    r"(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bNo)(?<!\b[A-Z])\.\s+(?=[A-Z0-9\"'-])",
                    block,
                )
                sents = []
                for s in parts:
                    s = s.strip()
                    if not s:
                        continue
                    if not s.endswith((".", "!", "?")):
                        s += "."
                    if len(re.findall(r"[A-Za-z0-9]+", s)) > 1:
                        sents.append(s)

                merged = []
                i = 0
                while i < len(sents):
                    if i + 1 < len(sents) and len(re.findall(r"[A-Za-z0-9]+", sents[i])) <= 4:
                        merged.append(f"{sents[i]} {sents[i + 1]}")
                        i += 2
                    else:
                        merged.append(sents[i])
                        i += 1

                for j, sent in enumerate(merged):
                    units = [sent] if j == 0 else [sent, f"{merged[j - 1]} {sent}"]
                    for u in units:
                        if clean_title and title_words and not any(tw in u.lower() for tw in title_words):
                            ctx = f"{clean_title}: {u}"
                        else:
                            ctx = u
                        if ctx not in seen:
                            seen.add(ctx)
                            windows.append((u, ctx))
        return windows

    import re
from typing import List, NamedTuple, Set

# Verdict labels
SUPPORTED = "S"
NOT_SUPPORTED = "NS"

# Maximum number of pruned candidates passed to the entailment model
MAX_CANDIDATES = 5

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_PARENTHETICAL_PATTERN = re.compile(r"\(.*?\)")


class _Candidate(NamedTuple):
    """A premise window scored by lexical overlap with the fact.

    Field order defines sort priority: higher overlap first, then shorter
    raw sentence (stored as a negative length), then the context string.
    """
    overlap: float
    neg_length: int
    premise: str


def _stemmed_tokens(self, text: str) -> Set[str]:
    """Lowercase, tokenise and stem `text`, returning the set of stems."""
    return {
        self.word_checker.stemmer.stem(token)
        for token in _TOKEN_PATTERN.findall(text.lower())
    }


def predict(self, fact: str, passages: List[dict]) -> str:
    """Classify whether `fact` is supported by the given passages.

    Pipeline:
        1. Extract content words from the fact, excluding tokens that
           appear in the primary passage title (they carry little signal).
        2. Prune candidate premise windows by lexical overlap with the fact.
        3. Run the entailment model on the top-ranked candidates only.

    Args:
        fact: The claim to verify.
        passages: Retrieved passages, each a dict with at least a "title"
            key; the first passage is treated as the primary document.

    Returns:
        "S" if any candidate premise entails the fact, otherwise "NS".
    """
    if not passages:
        return NOT_SUPPORTED

    fact_words = self._extract_fact_words(fact, passages[0].get("title", ""))
    if not fact_words:
        return NOT_SUPPORTED

    candidates = self._rank_candidates(fact_words, passages)

    for candidate in candidates[:MAX_CANDIDATES]:
        probability = self.ent_model.check_entailment(candidate.premise, fact)
        if probability >= self.ent_threshold:
            return SUPPORTED

    return NOT_SUPPORTED


def _extract_fact_words(self, fact: str, title: str) -> Set[str]:
    """Return content-word stems of `fact`, excluding title tokens.

    Falls back to the unfiltered preprocessing if excluding the title
    leaves no words.
    """
    clean_title = _PARENTHETICAL_PATTERN.sub(" ", title)
    title_tokens = self._stemmed_tokens(clean_title)

    words = self.word_checker.preprocess(fact, exclude_tokens=title_tokens)
    return words or self.word_checker.preprocess(fact)


def _rank_candidates(
    self, fact_words: Set[str], passages: List[dict]
) -> List[_Candidate]:
    """Score premise windows by overlap with `fact_words`, best first.

    Windows whose overlap falls below `self.prune_threshold`, or is zero,
    are discarded to avoid unnecessary entailment-model calls.
    """
    candidates: List[_Candidate] = []

    for raw_sentence, context in self.get_windows(passages):
        sentence_words = self._stemmed_tokens(raw_sentence)
        overlap = len(fact_words & sentence_words) / len(fact_words)

        if overlap > 0 and overlap >= self.prune_threshold:
            candidates.append(_Candidate(overlap, -len(raw_sentence), context))

    candidates.sort(reverse=True)
    return candidates
    
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
