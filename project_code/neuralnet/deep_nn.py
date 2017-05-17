'''
Created on Apr 21, 2017

@author: Amarnath
'''

from gensim import corpora
from gensim.models.ldamodel import LdaModel as LDA
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument as TD

def doc2vec_features(documents_hash, feature_count = 1000):
    business_ids = documents_hash.keys()
    documents = [documents_hash[b_id] for b_id in business_ids]
    tagged_documents = []
    for i, doc in enumerate(documents):
        tdoc = TD(doc.split(), tags = [i])
        tagged_documents.append(tdoc)

    doc2vec_model = Doc2Vec(tagged_documents, size = feature_count, iter = 5)
    doc2vec_features = [doc2vec_model.docvecs[i] for i in range(len(documents))]

    return doc2vec_features, business_ids, doc2vec_model

#def get_features_from_pretrained_model(documents, doc2vec_model):
#    doc1words = documents[0].split()
#    new_vector = doc2vec_model.infer_vector(doc1words)
#    print new_vector