
import numpy as np
import pandas as pd
from sklearn import base
from sklearn.feature_extraction import DictVectorizer # for One_Hot_Encoder



''' ----------------
Functions used in One_Hot_Encoder
----------------- '''
def Value_To_Dict(val):
    return {val:1}

def List_To_Dict(the_list):
    return {category:1 for category in the_list}
    
def Flatten_Dict(d, prekey = ''):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, bool) and v:
            flat_dict.update({prekey+'_'+k:1})
        elif isinstance(v, str):
            flat_dict.update({prekey+'_'+k+'_'+v:1})
        elif isinstance(v, dict):
            flat_dict.update(Flatten_Dict(v, prekey=prekey+'_'+k))
    return flat_dict

def flatten(structure, key="", path="", flattened=None):
    if flattened is None:
        flattened = {}
    if type(structure) not in(dict, list):
        flattened[((path + "_") if path else "") + key] = structure
    elif isinstance(structure, list):
        for i, item in enumerate(structure):
            flatten(item, "%d" % i, path + "_" + key, flattened)
    else:
        for new_key, value in structure.items():
            flatten(value, new_key, path + "_" + key, flattened)
    return flattened
''' -------------------
Converts a feature column values into a One-Hot Encoding matrix. If
feature values are lists or (nested) dicts, a column for each list 
entry or dict (sub)key is created.
Inputs: colnames is a string of the column name
        value_type is the type (value, list or dict) of feature values
        sparse indicates whether the matrix is sparse
Dependencies: sklearn.feature_extraction.DictVectorizer
              sklearn.base
------------------- '''
class One_Hot_Encoder(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, colnames, value_type = 'value', sparse = True):
        if value_type == 'value':
            self.apply_function_ = Value_To_Dict
        elif value_type == 'list':
            self.apply_function_ = List_To_Dict
        elif value_type == 'dict':
            self.apply_function_ = flatten
        self.colnames_ = colnames
        self.dv_ = DictVectorizer(sparse = sparse)

    def fit(self, X, y = None):
        self.dv_.fit(X[self.colnames_].apply(self.apply_function_))
        return self

    def transform(self, X):
        return self.dv_.transform(X[self.colnames_].apply(self.apply_function_))






''' -------------------
Selects and returns the specified column(s)
Inputs: colnames is a list of column(s) to select
Dependencies: sklearn.base
------------------- '''
class Column_Selector(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, colnames):
        self.colnames_ = colnames

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return pd.DataFrame(X[self.colnames_])
