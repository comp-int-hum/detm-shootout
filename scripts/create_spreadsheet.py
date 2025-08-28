import argparse
import math
import logging
import gzip
import json
import sys
import warnings
import ruptures
import pickle
import numpy
import torch
import pandas
from nltk.stem import PorterStemmer
from cltk.lemmatize.grc import GreekBackoffLemmatizer
from cltk.stops import grc
from nltk.corpus import stopwords
from cltk.alphabet.grc.grc import normalize_grc
from cltk.alphabet.lat import remove_accents
import re
import unicodedata

warnings.simplefilter("ignore")

logger = logging.getLogger("create_figures")

"""
Topics over time
Topic correlation
Words by topic-shift
Topics by embedding similarity plus temporal tradeoff
Topics by coherence

"""

cache = {}
def normalize(w):
    chars = []
    for c in w:
        cache[c] = cache.get(c, unicodedata.lookup(re.sub(r" WITH .*$", "", unicodedata.name(c))))
        chars.append(cache[c])
    return "".join(chars).lower()
#return remove_accents(normalize_grc(w.lower())).rstrip("-")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_data", dest="model_data", help="Model data (distributions, lookups, etc)", required=True)
    parser.add_argument("--human_annotation", dest="human_annotation", help="Human annotation of the topics")
    parser.add_argument("--documents", dest="documents", help="Documents of interest")
    parser.add_argument("--embeddings", dest="embeddings", help="Non-contextual word embeddings")
    parser.add_argument("--output", dest="output", help="Name of excel file for resulting spreadsheets", required=True)
    parser.add_argument("--num_top_words", dest="num_top_words", type=int, default=10)
    parser.add_argument("--stem", dest="stem", default=False, action="store_true", help="Stem words and sum probabilities")
    parser.add_argument("--stop", dest="stop", default=False, action="store_true", help="Ignore stop-words")
    parser.add_argument("--language", dest="language", help="Three-letter language code")
    parser.add_argument("--glosses", dest="glosses", help="File of word-glosses")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)

    default_stemmer = PorterStemmer()
    stemmers = {
        "grc" : GreekBackoffLemmatizer()
    }

    default_stops = stopwords.words("english")
    stops = {
        "grc" : [normalize(w) for w in grc.STOPS]
    }
    
    # load model distributions
    with open(args.model_data, "rb") as ifd:
        data = pickle.loads(ifd.read())

    # read in topic labels, if provided
    topic_labels = []
    if args.human_annotation:
        with pandas.ExcelFile(args.human_annotation) as ifd:
            header = None
            for i, row in enumerate(ifd.book["Topics"].rows):
                if i == 0:
                    header = [c.value for c in row]
                else:
                    rowdict = {k : v for k, v in zip(header, [c.value for c in row])}
                    topic_labels.append(rowdict["Label"])

    # load documents, if provided
    documents = []
    if args.documents:
        with gzip.open(args.documents, "rt") as ifd:
            pass

    # load word embeddings, if provided
    embeddings = {}
    if args.embeddings:
        with gzip.open(args.embeddings, "rb") as ifd:
            pass

    # load glosses, if provided
    glosses = {}
    if args.glosses:
        with gzip.open(args.glosses, "rt") as ifd:
            for line in ifd:
                toks = line.strip().split("\t")
                if len(toks) > 2:
                    word, word2, gloss = toks[:3]
                    word = normalize(word)
                    word2 = normalize(word2)
                    glosses[word.rstrip("-")] = gloss
                    glosses[word2.rstrip("-")] = gloss

    words = numpy.array([normalize(data["index_to_word"][i].lower()) for i in range(len(data["index_to_word"]))])
    stems = {}
    stem_to_word_indices = {}
    for index, word in data["index_to_word"].items():
        if args.stop and (not args.language and word in default_stops) or normalize(word) in stops.get(args.language, []):
            continue
        stem = normalize(word if not args.stem else default_stemmer.stem(word) if args.language not in stemmers else stemmers[args.language].lemmatize([word])[0][1])
        stem_to_word_indices[stem] = stem_to_word_indices.get(stem, [])
        stem_to_word_indices[stem].append(index)
    stem_to_index = {stem : i for i, stem in enumerate(stem_to_word_indices.keys())}
    index_to_stem = {i : stem for stem, i in stem_to_index.items()}

    stems = numpy.array([index_to_stem[i] for i in range(len(index_to_stem))])
    print(stems.shape)
    
    # 
    with pandas.ExcelWriter(args.output, mode="w") as ofd:
        
        P_wbt = torch.from_numpy(numpy.transpose(data["topic_window_word"], (2, 1, 0))) # P(w|b,t), word x bucket x topic (.sum(0) == 1.0)
        

        P_tbw = torch.permute(P_wbt / torch.unsqueeze(P_wbt.sum(2), 2), (2, 1, 0)) # topic x bucket x word
        P_tbs = torch.zeros(size=(P_tbw.shape[0], P_tbw.shape[1], len(stem_to_index)))
        P_sbt = torch.zeros(size=(len(stem_to_index), P_wbt.shape[1], P_wbt.shape[2]))
        
        for stem, i in stem_to_index.items():
            #ptbs[:, :, i] = ptbw[:, :, stem_to_word_indices[stem]].sum(2)
            P_sbt[i, :, :] = P_wbt[stem_to_word_indices[stem], :, :].sum(0)
            
        # this isn't normalized
        tb = torch.from_numpy(numpy.transpose(data["window_topic"], [1, 0]))
        
        # so, normalize it
        P_tb = tb / tb.sum(0).T # P(t|b)
        P_bt = tb.T / tb.sum(1) # P(b|t)

        topics = []
        for tid in range(P_wbt.shape[2]):
            aP_wt = P_wbt[:, :, tid].sum(1) / P_wbt.shape[1]
            top_indices = torch.argsort(aP_wt, dim=0, descending=True)[0:args.num_top_words]
            topic = {
                "Topic" : tid + 1,
                #"Entropy" : torch.distributions.Categorical(pbt[:, tid]).entropy().item(),
                "Likeliest Words" : ", ".join([glosses[w] if w in glosses else w for w in words[top_indices]])
            }
            print(topic)
            #for bid in range(pwbt.shape[1]):
            #    topic[data["index_to_window"][bid]] = ptb[tid][bid].item()

            topics.append(topic)

        pandas.DataFrame(topics).to_excel(ofd, sheet_name="Topics", index=False, float_format="%.4f")
