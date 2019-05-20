import nltk
import pickle
import scipy

nltk.download('conll2000')

from nltk.corpus import conll2000
print(conll2000.chunked_sents('train.txt')[99])

tree = conll2000.chunked_sents('train.txt')
print(tree)

test_sentences = conll2000.chunked_sents('test.txt')
train_sentences = conll2000.chunked_sents('train.txt')
for sent in train_sentences:
  print (nltk.chunk.tree2conlltags(sent))
  break

class ConsecutiveChunkTagger(nltk.TaggerI):
  
  def __init__(self, train_sentences):
    train_set = []
    for tagged_sent in train_sentences:
      untagged_sent = nltk.tag.untag(tagged_sent)
      history = []
      for i, (word, tag) in enumerate(tagged_sent):
        featureset = chunk_features(untagged_sent, i, history)
        train_set.append( (featureset, tag) )
        history.append(tag)
    print("Training...")
    self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='GIS', trace=0)
    print("Training Done, Evaluating...")
  def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = chunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveChunker(nltk.ChunkParserI):
  
  def __init__(self, train_sentences):
    tagged_sents = [[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)] for sent in train_sentences]
    self.tagger = ConsecutiveChunkTagger(tagged_sents)

  def parse(self, sentence):
    tagged_sents = self.tagger.tag(sentence)
    conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
    return nltk.chunk.conlltags2tree(conlltags)

def chunk_features(sentence, i, history):
  word, pos = sentence[i]
  if i == 0:
    prev3pos, prev2pos, prev1word, prev1pos = "<START>", "<START>","<START>", "<START>"
  elif i==1:
    prev1word, prev1pos = sentence[i-1]
    prev3pos, prev2pos = "<START>", "<START>"
  elif i==2:
    prev1word, prev1pos = sentence[i-1]
    prev2word, prev2pos = sentence[i-2]
    prev3pos = "<START>"
  else:
    prev1word, prev1pos = sentence[i-1]
    prev2word, prev2pos = sentence[i-2]
    prev3word, prev3pos = sentence[i-3]
  if i==len(sentence)-1:
    nextword, nextpos, next2pos = "<START>", "<START>","<START>"
  elif i==len(sentence)-2:
    nextword, nextpos = sentence[i+1]
    next2pos = "<START>"
  else:
    nextword, nextpos = sentence[i+1]
    next2word, next2pos = sentence[i+2]
  return {"pos": pos, "word": word, "prev1word":prev1word,"nextword": nextword, "nextpos": nextpos, "prev1pos": prev1pos,
   "prev2pos": prev2pos, "prev3pos": prev3pos,"next2pos": next2pos, "prev2pos+pos": "%s+%s" % (prev2pos, pos),
   "pos+nextpos": "%s+%s" % (pos, nextpos), "prev3pos+pos": "%s+%s" % (prev3pos, pos),
   "prev3pos+prev2pos": "%s+%s" % (prev3pos, prev2pos), "pos+next2pos": "%s+%s" % (pos, next2pos),
   "prev1pos+pos+nextpos": "%s+%s+%s" % (prev1pos, pos, nextpos), "pos+nextpos+next2pos": "%s+%s+%s" % (pos, nextpos, next2pos),
   "prev2pos+prev1pos+pos+nextpos": "%s+%s+%s+%s" % (prev2pos, prev1pos, pos, nextpos)}
chunker = ConsecutiveChunker(train_sentences)
trained_model = 'trained_model.pkl'
f = open(trained_model,'wb')
pickle.dump(chunker, f)
f.close()
print(chunker.evaluate(test_sentences))