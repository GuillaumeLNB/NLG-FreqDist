import nltk
import random
from nltk import word_tokenize


class NgramGenerator:
    def __init__(self, text: str):
        self.text = text
        self.tokens = word_tokenize(text)

        # 2-grams
        self.bigrams = nltk.bigrams(self.tokens)
        self.cfreq_2gram = nltk.ConditionalFreqDist(self.bigrams)
        self.cprob_2gram_laplace = nltk.ConditionalProbDist(
            self.cfreq_2gram, nltk.LaplaceProbDist
        )
        # 3-grams
        self.trigrams = nltk.trigrams(self.tokens)
        self.cfreq_3gram = nltk.ConditionalFreqDist(
            ((w1, w2), w3) for w1, w2, w3 in self.trigrams
        )

        self.cprob_3gram_laplace = nltk.ConditionalProbDist(
            self.cfreq_3gram, nltk.LaplaceProbDist
        )
        # 4-grams
        self.fourgrams = nltk.ngrams(self.tokens, 4)
        self.cfreq_4gram = nltk.ConditionalFreqDist(
            ((w1, w2, w3), w4) for w1, w2, w3, w4 in self.fourgrams
        )
        self.cprob_4gram_laplace = nltk.ConditionalProbDist(
            self.cfreq_4gram, nltk.LaplaceProbDist
        )

    def _check_words_are_in_text(self, words):
        for word in words:
            if word not in self.tokens:
                raise ValueError(f"'{word}' is not in the text")

    def _get_model(self, number: int):
        try:
            return getattr(self, f"cfreq_{number}gram")
        except AttributeError:
            raise ValueError(f"Model {number} cfreq_{number}gram not implemented")

    def generate_text_deterministic(self, history, model, size=100):
        history_words = word_tokenize(history)
        self._check_words_are_in_text(history_words)
        model = self._get_model(model)

        for _ in range(size):
            next_word = model[(history_words[-2], history_words[-1])].max()
            history_words.append(next_word)
        return " ".join(history_words)

    def _get_rand_nbest(self, hist, model, n):
        """
        Returns a random word among the n bests according to the model
        Args:
            hist (tuple): the words preceding (tuple of 2 words for a 3gram model)
            model (ConditionalFreqDist): ngram model
            n (int): n best words
        Returns:
            str: random word among the n bests
        """
        bests = sorted(
            # model[hist].samples(),
            model[hist].most_common(100),
            key=lambda sample: model[hist].prob(sample),
            reverse=True,
        )
        if len(bests) > n:
            n_best = bests[:n]
            word = n_best[random.randint(0, n - 1)]
        else:
            n_best = bests
            word = n_best[random.randint(0, (len(n_best) - 1))]
        return word

    def generate_text(self, history, model: int, size=100):
        history_words = word_tokenize(history)
        self._check_words_are_in_text(history_words)
        model = self._get_model(model)
        for _ in range(size):
            next_word = self._get_rand_nbest(
                (history_words[-2], history_words[-1]), model, 15
            )
            history_words.append(next_word)
        return " ".join(history_words)
