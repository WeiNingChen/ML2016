from nltk.corpus import reuters
from math import log 
from nltk import WordNetLemmatizer
wnl=WordNetLemmatizer()

_DN = float(len(reuters.fileids()))
_RTS = {k:[wnl.lemmatize(w.lower()) for w in reuters.words(k)] for k in reuters.fileids()}

def idf(w):
    w = wnl.lemmatize(w.lower())
    return log( _DN / sum([float(w in _RTS[k]) for k in _RTS.keys() ]) , 2)

def tf(w,f):
    w = wnl.lemmatize(w.lower())
    return sum([float(w == x) for x in _RTS[f] ]) 

def tf_idf(w,f):
    return tf(w,f)*idf(w)

def run(f):
    word_tfidf = [(w,tf_idf(w,f)) for w in set(_RTS[f])] 
    for w,t in sorted(word_tfidf, key = lambda x : x[1], reverse=True):
        print "%-15s %.10f"%(w,t)
