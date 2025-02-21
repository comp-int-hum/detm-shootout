import numpy as np
import logging
import gzip
import json
import argparse
import torch
from detm import Corpus, apply_model, load_embeddings
import torch
import umap
from detm import evaluate_coherence, evaluate_topic_diversity, original_detm_evaluation
import gensim.downloader as api

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


def temporal_coherence_aligned(topic_vectors):
    coherence_scores = []
    for i in range(len(topic_vectors) - 1):
        similarities = cosine_similarity(topic_vectors[i], topic_vectors[i + 1])
        same_topic_similarities = np.diag(similarities)
        coherence_score = np.mean(same_topic_similarities)
        coherence_scores.append(coherence_score)
    return coherence_scores

def temporal_coherence(topic_vectors):
    coherence_scores = []
    for i in range(len(topic_vectors) - 1):
        similarity_matrix = cosine_similarity(topic_vectors[i], topic_vectors[i + 1])
        coherence_score = np.mean(np.max(similarity_matrix, axis=1))
        coherence_scores.append(coherence_score)
    return coherence_scores

def topic_diversity(topic_vectors):
    diversity_scores = []
    for topics in topic_vectors:
        similarity_matrix = cosine_similarity(topics, topics)
        np.fill_diagonal(similarity_matrix, 0)
        diversity_score = 1 - np.mean(similarity_matrix)
        diversity_scores.append(diversity_score)
    return diversity_scores

logger = logging.getLogger("evaluate")
logger.setLevel(logging.WARNING)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument("--embeddings", dest="embeddings", help="Embeddings file")
    parser.add_argument("--reference_corpus", dest="reference_corpus", help="Reference corpus")
    parser.add_argument("--topn", dest="topn", type=int, default=25, help="Number of top words to consider for coherence")
    args = parser.parse_args()

    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    device = "cpu"

    print(args.embeddings)
    embeddings = load_embeddings(args.embeddings)

    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device(device), weights_only=False)
    
    model = model.to(device)
    model.eval()

    reference_corpus = Corpus()
    reference_corpus_text = []
    with gzip.open(args.reference_corpus, "rt") as ifd:
        for i, line in enumerate(ifd):
            data = json.loads(line)
            reference_corpus_text.extend(data["text"])
            reference_corpus.append(data)
    coherence_scores = {}

    subdocs, times = reference_corpus.filter_for_model(model, "text", "year")

    original_diversity, original_coherence, original_quality = original_detm_evaluation(model, subdocs)

    topic_words = model.get_topic_words(args.topn)
    for coherence_type in ["c_v", "c_uci", "c_npmi", "u_mass", "c_w2v"]:
        if coherence_type == "c_w2v":
            coherence = evaluate_coherence(topics=topic_words, coherence_measure=coherence_type, keyed_vectors=embeddings, text=reference_corpus_text, topn=args.topn) #, window_size=200)
        else:
            coherence = evaluate_coherence(topics=topic_words, coherence_measure=coherence_type, text=reference_corpus_text, topn=args.topn) #, window_size=200)
        coherence_scores[coherence_type] = coherence
    
    diversity_scores = {}
    for diversity_measure in ['proportion_unique_words', 'irbo', 'word_embedding_irbo', 'pairwise_jaccard_diversity', 'pairwise_word_embedding_distance', 'centroid_distance']:
        diversity = evaluate_topic_diversity(model, diversity_measure, embedding=embeddings, topn=args.topn)
        diversity_scores[diversity_measure] = diversity

    alphas = model.topic_distributions().detach().cpu().numpy()

    scores = {
        "coherence_scores" : coherence_scores,
        "diversity_scores" : diversity_scores,
        "original_scores" : {
            "diversity" : original_diversity,
            "coherence" : original_coherence,
            "quality" : original_quality
        }
    }
    with open(args.output, "wt") as ofd:
        ofd.write(json.dumps({"evaluation" : scores}) + "\n")
