import nltk
import random
from nltk import word_tokenize

from unidecode import unidecode
import pickle
import sys

fic =unidecode(open("ric.txt").read().lower())
unidecode(open("../data/text_DEMOCRATIE.txt", encoding="utf-8").read().lower()[:203])


words_nom = word_tokenize(fic)
words_nom=[w for w in words_nom if w and w!='``']
print(len(words_nom))
# cfreq_2gram_nom = nltk.ConditionalFreqDist(nltk.bigrams(words_nom))

# cprob_2gram_mle_nom = nltk.ConditionalProbDist(cfreq_2gram_nom, nltk.MLEProbDist)

# cprob_2gram_laplace_nom = nltk.ConditionalProbDist(cfreq_2gram_nom, nltk.LaplaceProbDist)

trigrams_nom = nltk.trigrams(words_nom)
cfreq_3gram_nom = nltk.ConditionalFreqDist(((w1, w2), w3) for w1, w2, w3 in trigrams_nom)


cprob_3gram_laplace_nom = nltk.ConditionalProbDist(cfreq_3gram_nom, nltk.LaplaceProbDist)

pickle.dump(cprob_3gram_laplace_nom, open("../models/cprob_3gram_laplace_nom.pkl", 'wb'))

# cprob_3gram_laplace_nom=pickle.load(open("../models/cprob_3gram_laplace_nom.pkl", "br"))

############ print(cprob_3gram_laplace_nom[('at', 'the')].max())

# def generate_text(history, model, size=100):
#     history_words = history.split(' ')
#     for i in range(size):
#         next_word = model[(history_words[-2], history_words[-1])].max()
#         history_words.append(next_word)
#     return " ".join(history_words)


# print(generate_text("je pense", cprob_3gram_laplace_nom, 40), end="\n"*2)

# def get_rand_nbest(hist, model, n):
#     """
#     Returns a random word among the n bests according to the model
#     Args:
#         hist (tuple): the words preceding (tuple of 2 words for a 3gram model)
#         model (ConditionalFreqDist): ngram model
#         n (int): n best words
#     Returns:
#         str: random word among the n bests
#     """
#     bests = sorted(model[hist].samples(), key=lambda sample: model[hist].prob(sample), reverse=True)
#     if len(bests) > n:
#         n_best = bests[:n]
#         word = n_best[random.randint(0, n-1)]
#     else:
#         n_best = bests
#         word = n_best[random.randint(0, (len(n_best)-1))]
#     return word

# def generate_text(history, model, size=100, n=3):
#     history_words = history.split(' ')
#     if len(history_words) < n-1:
#         raise ValueError("history has only {} values, {} needed".format(len(history_words), n))
#     for i in range(size):
#         next_word = get_rand_nbest((history_words[-2], history_words[-1]), model, 15)
#         history_words.append(next_word)
#     return " ".join(history_words)


# n=sys.argv[1] if sys.argv else 30
# for _ in range(10):
#     print(generate_text("il faut que le ric", cprob_3gram_laplace_nom, n),end='\n'*2)