import random
import logging
import gzip
import json
import argparse
import numpy
import torch
from detm import Corpus, train_model, load_embeddings
import json

logger = logging.getLogger("train_model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", dest="train", help="Data file")
    parser.add_argument("--time_field", dest="time_field", help="")
    parser.add_argument("--content_field", dest="content_field", help="")
    parser.add_argument("--output", dest="output", help="File to save word_list to", required=True)
    parser.add_argument("--min_word_count", dest="min_word_count", type=int, default=0, help="Words occuring less than this number of times throughout the entire dataset will be ignored")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None, help="Specify a random seed (for repeatability)")
    parser.add_argument("--max_word_proportion", dest="max_word_proportion", type=float, default=1.0, help="Words occuring in more than this proportion of documents will be ignored (probably conjunctions, etc)")    
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    
    if args.random_seed:
        random.seed(args.random_seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)

    corpus = Corpus()

    with gzip.open(args.train, "rt") as ifd:
        for i, line in enumerate(ifd):
            corpus.append(json.loads(line))

    subdocs, times, word_list = corpus.get_filtered_subdocs(
        content_field=args.content_field,
        time_field=args.time_field,
        min_word_count=args.min_word_count,
        max_word_proportion=args.max_word_proportion,
    )
    
    # save word list
    with open(args.output, "w") as ofd:
        json.dump(word_list, ofd)