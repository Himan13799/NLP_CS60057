# Maximum Entropy Model

- Implementation of the paper [Koe00]
Rob Koeling, Chunking with Maximum Entropy Models. In: Proceedings of
CoNLL-2000 and LLL-2000, Lisbon, Portugal, 2000.


# RUN 


- To run in terminal enter 'python3 maxEntClassifier.py'

- it classifies the words into 22 possible chunk tags

- output is an object which is dumped into a pickle file 'trained_model.pkl'

- evaluation is done in the code itself using the `nltk.MaxentClassifier` for training and then evaluation purpose

- chunker.evaluate(test_sentences) where `test_sentences` is a tree

- an image of evaluated maxEnt model is included in this directory