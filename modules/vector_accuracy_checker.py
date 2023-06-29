"""
MIT License

Copyright (c) 2018 NLX-Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code builds a word2vec model based on the test-set (the created embeddings)
Then model_accuracy is checked based on the comparison berween the existing accuracy_test_sets in Gensim

2017     Chakaveh.saedi@di.fc.ul.pt


help:
https://radimrehurek.com/gensim/models/keyedvectors.html
"""


import os
import logging
import gensim
from typing import List, Union

from modules.input_output import *


def vector_accuracy(eval_set_paths: List[str], emb_path: str, output_dir: str, lang: str):
    print("\n* Checking accuracy")
    log_file = os.path.join(output_dir, "accuracy_results")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename=log_file, filemode="w"
    )
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # to build the model based on the created embeddings and then compare it to the reference
    # load and evaluate
    _, file_extension = os.path.splitext(emb_path)
    model = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=(file_extension == ".bin"))
    for eval_set_path in eval_set_paths:
        print("eval_set_path: %s-----------------------------" % eval_set_path)
        if "questions-words" in eval_set_path:
            model.accuracy(eval_set_path, restrict_vocab=None)
        else:
            model.evaluate_word_pairs(eval_set_path)
