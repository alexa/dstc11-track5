import numpy as np

from nltk import bigrams as get_bigrams
from nltk import trigrams as get_trigrams
from nltk import word_tokenize, ngrams
from collections import Counter

from rouge_score import rouge_scorer
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric

from .data import normalize


def get_fourgrams(sequence, **kwargs):
    """
    Return the 4-grams generated from a sequence of items, as an iterator.

    :param sequence: the source data to be converted into 4-grams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 4, **kwargs):
        yield item


class Metric:
    def __init__(self):
        self.is_single = True
        self.reset()

    def reset(self):
        pass

    def update(self, output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


class DataCacheMetric(Metric):
    def __init__(self):
        self.refs = []
        self.preds = []
        super(DataCacheMetric, self).__init__()

    def reset(self):
        self.refs = []
        self.preds = []

    def update(self, output):
        hypothesis, reference = output
        assert isinstance(hypothesis, str)
        assert isinstance(reference, str)
        self.preds.append(hypothesis)
        self.refs.append(reference)

    def compute(self):
        return len(self.preds)

    def name(self):
        return "Data Count"


class UnigramMetric(Metric):
    def __init__(self, metric):
        self._score = None
        self._count = None
        if metric.lower() not in ["recall", "precision"]:
            raise ValueError("mertic should be either 'recall' or 'precision', got %s" % metric)
        self.metric = metric.lower()
        super(UnigramMetric, self).__init__()

    def reset(self):
        self._score = 0
        self._count = 0
        super(UnigramMetric, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        hyp_tokens = normalize(hypothesis).split()
        ref_tokens = normalize(reference).split()

        common = Counter(ref_tokens) & Counter(hyp_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            score = 0
        else:
            if self.metric == "precision":
                score = 1.0 * num_same / len(hyp_tokens)
            else:
                assert self.metric == "recall"
                score = 1.0 * num_same / len(ref_tokens)

        self._score += score
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("Unigram metrics must have at least one example before it can be computed!")
        return self._score / self._count

    def name(self):
        return "Unigram{:s}".format(self.metric.capitalize())


class NGramDiversity(Metric):
    def __init__(self, n=1):
        self._n = n
        self._diversity = None
        self._count = None

        if self._n not in [1, 2, 3, 4]:
            raise ValueError("NGramDiversity only supports n=1 (unigrams), n=2 (bigrams),"
                             "n=3 (trigrams) and n=4 (4-grams)!")

        self.ngram_func = {
            1: lambda x: x,
            2: get_bigrams,
            3: get_trigrams,
            4: get_fourgrams
        }[self._n]

        super(NGramDiversity, self).__init__()

    def reset(self):
        self._diversity = 0
        self._count = 0
        super(NGramDiversity, self).reset()

    def update(self, output):
        hypothesis, _ = output

        if hypothesis is None:
            diversity = 0
        else:
            diversity = 0
            output_tokens = word_tokenize(hypothesis)
            denominator = float(len(output_tokens))

            if denominator != 0.0:
                ngrams = set(list(self.ngram_func(output_tokens)))
                diversity = len(ngrams) / denominator

        self._diversity += diversity
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("NGramDiversity must consume at least one example before it can be computed!")
        return self._diversity / self._count

    def name(self):
        return "{:d}GramDiversity".format(self._n)


class CorpusNGramDiversity(Metric):
    def __init__(self, n=1):
        self._n = n

        self._ngrams = None
        self._token_count = None

        if self._n not in [1, 2, 3, 4]:
            raise ValueError("CorpusNGramDiversity only supports n=1 (unigrams), n=2 (bigrams),"
                             "n=3 (trigrams) and n=4 (4-grams)!")
        self.ngram_func = {
            1: lambda x: x,
            2: get_bigrams,
            3: get_trigrams,
            4: get_fourgrams
        }[self._n]

        super(CorpusNGramDiversity, self).__init__()

    def reset(self):
        self._ngrams = set()
        self._token_count = 0
        super(CorpusNGramDiversity, self).reset()

    def update(self, output):
        hypothesis, _ = output
        if isinstance(hypothesis, str) and hypothesis:
            output_tokens = word_tokenize(hypothesis)

            ngrams = list(self.ngram_func(output_tokens))
            self._ngrams.update(ngrams)
            self._token_count += len(output_tokens)

    def compute(self):
        if self._token_count == 0:
            raise ValueError("CorpusNGramDiversity must consume at least one example before it can be computed!")

        return len(self._ngrams) / self._token_count

    def name(self):
        return "Corpus{:d}GramDiversity".format(self._n)


class LENGTH(DataCacheMetric):
    def __init__(self):
        self._len = []
        super(LENGTH, self).__init__()

    def reset(self):
        self._len = []

    def update(self, output):
        hypothesis, _ = output
        self._len.append(len(hypothesis.split()))

    def compute(self):
        if len(self._len) == 0:
            raise ValueError("LENGTH must have at least one example before it can be computed!")
        return sum(self._len) / len(self._len)

    def name(self):
        return "LENGTH"


class BLEU(DataCacheMetric):
    def __init__(self):
        super(BLEU, self).__init__()

    def compute(self):
        if len(self.preds) == 0:
            raise ValueError("BLEU-1 must have at least one example before it can be computed!")

        metric = BleuMetric()
        score = metric.evaluate_batch(self.preds, self.refs)
        return score['bleu']

    def name(self):
        return "BLEU"


class METEOR(DataCacheMetric):
    def __init__(self):
        super(METEOR, self).__init__()

    def compute(self):
        if len(self.preds) == 0:
            raise ValueError("METEOR must have at least one example before it can be computed!")
        metric = MeteorMetric()
        score = metric.evaluate_batch(self.preds, self.refs)
        return score['meteor'] * 100

    def name(self):
        return "METEOR"


class ROUGE(Metric):
    def __init__(self):
        self.rouge_type = ['rouge1', 'rouge2', 'rougeL', "rougeLsum"]
        self.scorer = rouge_scorer.RougeScorer(self.rouge_type, use_stemmer=True)
        self._rouge = None
        self._count = None
        super(ROUGE, self).__init__()
        self.is_single = False

    def reset(self):
        self._rouge = []
        self._count = 0
        super(ROUGE, self).reset()

    def update(self, output):
        hypothesis, reference = output
        rouge = self.scorer.score(reference, hypothesis)

        _rouge = [rouge[_rouge_type].fmeasure * 100 for _rouge_type in self.rouge_type]
        self._rouge.append(_rouge)
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("ROUGE-L must have at least one example before it can be computed!")
        return np.array(self._rouge).mean(axis=0).tolist()

    def name(self):
        return self.rouge_type
