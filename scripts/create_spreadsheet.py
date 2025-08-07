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

warnings.simplefilter("ignore")

logger = logging.getLogger("create_figures")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--data", dest="data", help="Data file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--num_top_words", dest="num_top_words", type=int, default=10)
    args = parser.parse_args()

    
    logging.basicConfig(level=logging.INFO)

    with open(args.input, "rb") as ifd:
        data = pickle.loads(ifd.read())
        
    with pandas.ExcelWriter(args.output, mode="w") as ofd:
        
        pwbt = torch.from_numpy(numpy.transpose(data["topic_window_word"], (2, 1, 0))) # P(w|b,t), word x bucket x topic (.sum(0) == 1.0)
        
        words = numpy.array([data["index_to_word"][i] for i in range(len(data["index_to_word"]))])

        ptbw = torch.permute(pwbt / torch.unsqueeze(pwbt.sum(2), 2), (2, 1, 0)) # topic x bucket x word

        # this isn't normalized
        tb = torch.from_numpy(numpy.transpose(data["window_topic"], [1, 0]))
        
        # so, normalize it
        ptb = tb / tb.sum(0).T # P(t|b)
        pbt = tb.T / tb.sum(1) # P(b|t)

        topics = []
        for tid in range(pwbt.shape[2]):
            
            a_pwt = pwbt[:, :, tid].sum(1) / pwbt.shape[1]
            top_indices = torch.argsort(a_pwt, dim=0, descending=True)[0:args.num_top_words]
            topic = {
                "Topic" : tid + 1,
                "Entropy" : torch.distributions.Categorical(pbt[:, tid]).entropy().item(),
                "Likeliest Words" : ", ".join([w for w in words[top_indices]])
            }
            for bid in range(pwbt.shape[1]):
                topic[data["index_to_window"][bid]] = ptb[tid][bid].item()

            topics.append(topic)

        pandas.DataFrame(topics).to_excel(ofd, sheet_name="Topics", index=False, float_format="%.4f")
