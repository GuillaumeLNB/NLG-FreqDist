import nltk
import random
from nltk import word_tokenize
import argparse

from unidecode import unidecode
import pickle
import sys

def get_rand_nbest(hist, model, n):
    """
    Returns a random word among the n bests according to the model
    Args:
        hist (tuple): the words preceding (tuple of 2 words for a 3gram model)
        model (ConditionalFreqDist): ngram model
        n (int): n best words
    Returns:
        str: random word among the n bests
    """
    bests = sorted(model[hist].samples(), key=lambda sample: model[hist].prob(sample), reverse=True)
    if len(bests) > n:
        n_best = bests[:n]
        word = n_best[random.randint(0, n-1)]
    else:
        n_best = bests
        word = n_best[random.randint(0, (len(n_best)-1))]
    return word

def generate_text(history, model, size=100, n=3):
    history_words = history.split(' ')
    if len(history_words) < n-1:
        raise ValueError("history has only {} values, {} needed".format(len(history_words), n))
    for i in range(size):
        next_word = get_rand_nbest((history_words[-2], history_words[-1]), model, 15)
        history_words.append(next_word)
    return " ".join(history_words)


def main(argv):
    ""
    parser = argparse.ArgumentParser(description="génère du txt", epilog="_")
    parser.add_argument('-p', dest='phrase', default="le ric" )
    parser.add_argument("-n", dest="num", type=int, help="le nombre de mots a générer", default=30)
    parser.add_argument("-t", dest="times", type=int, help="le nombre de fois", default=10)
    args=parser.parse_args(argv)

    # if len(argv)<1:
    #     parser.print_help()
    #     sys.exit(1)
    cprob_3gram_laplace_nom=pickle.load(open("../models/cprob_3gram_laplace_nom.pkl", "br"))

    for _ in range(args.times):
        print(generate_text(args.phrase, cprob_3gram_laplace_nom, args.num),end='\n'*2)


if __name__=="__main__":
    sys.exit(main(sys.argv[1:]))