'''
Build a recommender system for the Yelp dataset.
* Content-based filtering
* User-user collaborative filtering

----------
AUTHOR: CHIA YING LEE
DATE: 10 APRIL 2015
----------
'''

import pandas as pd
import ast
import numpy as np
from numpy.linalg import norm
from sklearn.pipeline import Pipeline, FeatureUnion
from src import transformers
from scipy.sparse import coo_matrix
import heapq


import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split
from _collections import defaultdict
# from src.load_data import df_business
# from src.load_data import df_review

print '**Loading data...'

# LOAD DATA FOR TYPE = dataset_type
fileheading = 'yelp_dataset_challenge/yelp_academic_dataset_'

def get_data(line, cols):
    d = json.loads(line)
    return dict((key, d[key]) for key in cols)
# 
# def CustomParser(data):
#     j1 = json.loads(data)
#     return j1

# Load business data
# cols = ('business_id', 'name')
# with open(fileheading + 'business.json') as f:
#     df_business = pd.DataFrame(get_data(line, cols) for line in f)
def string_to_dict(dict_string):
    # Convert to proper json format
#     print type(ast.literal_eval(dict_string))
#     print dict_string.__class__
    if dict_string is not np.nan:
        dict_string = ast.literal_eval(dict_string)
#     dict_string = dict_string.replace("[", "{").replace("]", "}").replace("'", "\"")
#     print ast.literal_eval(dict_string)
#     d = json.loads(dict_string)
#     dict = {}
#     print dict_string
# #     list = dict_string.split
#     d = json.loads(dict_string)
#     print dict_string.__class__
    return dict_string

df_business = pd.read_csv("yelp_dataset_challenge/southcarolina.csv")
df_business = df_business[df_business['business_id'].str.contains("#NAME") == False]
df_business = df_business.sort('business_id')
# print type(df_business['attributes'][0])
# df_business.join(df_business['attributes'].apply(json.loads).apply(pd.Series)) 
# df_business.attributes = df_business.attributes.astype(dict)
df_business.attributes = df_business.attributes.apply(string_to_dict)
print type(df_business['attributes'][0])
df_business.index = range(len(df_business))

# Load user data
# cols = ('user_id', 'name')
# with open(fileheading + 'user.json') as f:
#     df_user = pd.DataFrame(get_data(line, cols) for line in f)
df_user = pd.read_csv("yelp_dataset_challenge/sc_user.csv")
df_user = df_user.sort('user_id')
df_user.index = range(len(df_user))

# Load review data
# cols = ('user_id', 'business_id', 'stars')
# with open(fileheading + 'review.json') as f:
#     df_review = pd.DataFrame(get_data(line, cols) for line in f)
df_review = pd.read_csv("yelp_dataset_challenge/sc_review.csv")

data_load_time = datetime.now()
print 'Data was loaded at ' + data_load_time.time().isoformat()

# # Load data
# try:
#     data_load_time
# except NameError:
#     execfile('src/load_data.py')
# else:
#     print 'Data was loaded at ' + data_load_time.time().isoformat()

# Personalized recommendation for a specific user
user = '_5uUOcTyKK1pQYwAL-Xrnw'

# ----------------
# CONTENT BASED FILTERING
# ----------------

print '*** Using Content-based Filtering for Recommendation ***'
print '** Initializing feature extraction for user ' + user

print '***Extracting deep nn features from reviews***'

business_ids = df_review['business_id']
# biz_id_set = set(business_ids)
document_hash = defaultdict(str)
for i in range(len(df_review)):
    try:
        document_hash[business_ids[i]] += df_review['text'][i]
    except:
        continue
    
# biz_id_doc_set = set(document_hash.keys())
# biz_id_bad_set = biz_id_set - biz_id_doc_set
# df_business = df_business[~df_business['business_id'].isin(biz_id_bad_set)]
# print len(df_business)

# Extract features of each business: category, attribute, average rating
OHE_cat = transformers.One_Hot_Encoder('categories', 'list', sparse=False)
OHE_attr= transformers.One_Hot_Encoder('attributes', 'dict', sparse=False)
OHE_city= transformers.One_Hot_Encoder('city', 'value', sparse=False)
rating = transformers.Column_Selector(['stars'])
OHE_union = FeatureUnion([ ('cat', OHE_cat), ('attr', OHE_attr), ('city', OHE_city), ('rating', rating) ])
# OHE_union = OHE_attr
OHE_union.fit(df_business)

from deep_nn import doc2vec_features
deepnn_business_features, business_id_order, doc2vec_model = doc2vec_features(document_hash)

deepnn_business_features = np.array(deepnn_business_features)
print len(deepnn_business_features)

# 1) Pick 100 users with good no. of reviews (like > 15 reviews)
#            - Take for example: 1000 businesses in total. User has reviewed 100 businesses.
# print len(df_user)
df_user_sub = df_user[df_user['review_count'] > 20]
# print len(df_user_sub)
users_idx = [v for v in df_user_sub['user_id']]
users_idx = list(set(users_idx))
len(users_idx)

cosine_sim_avg_normal = 0.0
cosine_sim_avg_deep = 0.0
for itr, user_id in enumerate(users_idx[:100]):
    print '*&*&*&*&*&*&*&*&*&*' + str(itr) 
    # 2) For each user, construct a profile using only 80% of the businesses he reviews (80 businesses from example).
    reviewed_businesses = df_review.ix[df_review.user_id == user]
    reviewed_businesses_sub = reviewed_businesses.head(int(0.8*len(reviewed_businesses)))
    reviewed_businesses_unused = reviewed_businesses[~reviewed_businesses['business_id'].isin(reviewed_businesses_sub['business_id'])]
    reviewed_business_ids_unused = reviewed_businesses_unused['business_id']
    unused_bid_list = [v for v in reviewed_business_ids_unused]
    
    star_hash = {}
    for i in range(len(df_review)):
        if df_review['business_id'][i] in unused_bid_list:
            star_hash[df_review['business_id'][i]] = df_review['stars'][i]
    
    # print reviewed_business_ids_unused
    
    test_list = [float(v) for v in df_user.average_stars[df_user.user_id == user_id]]
    reviewed_businesses_sub['stars'] = reviewed_businesses_sub['stars'] - float(test_list[0])
    idx_reviewed_sub = [pd.Index(df_business.business_id).get_loc(b) for b in reviewed_businesses_sub.business_id]
    idx_unused_bizs = [pd.Index(df_business.business_id).get_loc(b) for b in reviewed_business_ids_unused]
    
    # print '**Creating profiles...'
    features_sub = OHE_union.transform(df_business.ix[idx_reviewed_sub])
    profile = np.matrix(reviewed_businesses_sub.stars) * features_sub 
    idx_in_deep = [business_id_order.index(b_id) for b_id in reviewed_businesses_sub.business_id]
    deepnn_features_reviewed = np.array([deepnn_business_features[i] for i in idx_in_deep])
    profile_deep = np.matrix(reviewed_businesses_sub.stars) * deepnn_features_reviewed
    
    idx_new = list(set(range(len(df_business.business_id))) - set(idx_reviewed_sub))
    idx_new = sorted(idx_new) 
    
    relevant_locs = [i for i, v in enumerate(idx_new) if v in idx_unused_bizs]
    
    # idx_new = [pd.Index(df_business.business_id).get_loc(b) for b in df_business.business_id if b not in reviewed_businesses_sub.business_id]
    # print idx_new[10].__class__
    # print len(idx_new)
    #     idx_new = [v for v in idx_new if isinstance(v, np.int64)]
    features = OHE_union.transform(df_business.ix[idx_new])
#   len(document_hash)
    # 3) Now compute the similarities with rest all (other than the 80% = 920 bizs) businesses and order the businesses in descending order of similarity
    similarity = np.asarray(profile * features.T) * 1./(norm(profile) * norm(features, axis = 1))
    deep_features_new_bizs = np.array([deepnn_business_features[i] for i in idx_new]) 
    similarity_deep = np.asarray(profile_deep * deep_features_new_bizs.T) * 1./(norm(profile_deep) * norm(deep_features_new_bizs, axis = 1))    
    #print 'Done'    
    deep_coefficient = 1
    combined_similarity = np.array([similarity[i] + deep_coefficient * (similarity_deep[i]) for i in range(len(similarity))])
    #print len(similarity[0])
    sorted_biz_idxs = sorted(range(len(idx_new)), key = lambda x : similarity[0][x], reverse = True)
    relevant_ranks = [i for i, v in enumerate(sorted_biz_idxs) if v in relevant_locs]
    relevant_biz_ids = [df_business.business_id[idx_new[v]] for v in relevant_locs]
    
    sorted_biz_idxs_deep = sorted(range(len(idx_new)), key = lambda x : combined_similarity[0][x], reverse = True)
    relevant_ranks_deep = [i for i, v in enumerate(sorted_biz_idxs_deep) if v in relevant_locs]
    
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity as cs
    
    star_vals = []
    mrr = 0
    for i, v in enumerate(relevant_ranks):
        star_vals.append(star_hash[relevant_biz_ids[i]])
        mrr += 1.0/(v + 1)
   
    mrr *= (1.0/len(relevant_ranks))
    print '----------------------------------'
    sim_val =  cs(star_vals, relevant_ranks)
    cosine_sim_avg_normal += sim_val   
    print mrr, sim_val
    
    mrr_deep = 0
    for i, v in enumerate(relevant_ranks_deep):
        mrr_deep += 1.0/(v + 1)
   
    mrr_deep *= (1.0/len(relevant_ranks_deep))
    sim_val_deep =  cs(star_vals, relevant_ranks_deep)
    cosine_sim_avg_deep += sim_val_deep   
    print mrr_deep, sim_val_deep        
    
    # 4) Now find the ranks of the 20% of the businesses which the user has reviewed,
    # but were not considered in constructing the profile. (20 businesses from example).
    
    
    
    # 5) Calculate MRR : 1/|Q| * SUM(1/rank_i). Average of inverted ranks for this specific user. (https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
    # 6) Take average of MRR for 100 users.
    # Output: recommend the most similar business
    #idx_recommendation = similarity.argmax()
    #idx_recommendation_deep = combined_similarity.argmax()


print '************************'
print cosine_sim_avg_normal/10.0
print cosine_sim_avg_deep/10.0
print '************************'
# Generate profile: weighted average of features for business she has reviewed
print '**Getting businesses...'
reviewed_businesses = df_review.ix[df_review.user_id == user]
print len(reviewed_businesses)
test_list = [float(v) for v in df_user.average_stars[df_user.user_id == user]]

reviewed_businesses['stars'] = reviewed_businesses['stars'] - float(test_list[0])
idx_reviewed = [pd.Index(df_business.business_id).get_loc(b) for b in reviewed_businesses.business_id]
print idx_reviewed

print '**Creating profiles...'
features = OHE_union.transform(df_business.ix[idx_reviewed])
profile = np.matrix(reviewed_businesses.stars) * features

# Creating deepnn features and profiles for this user  --> Need to do this for many users
idx_in_deep = [business_id_order.index(b_id) for b_id in reviewed_businesses.business_id]
deepnn_features_reviewed = np.array([deepnn_business_features[i] for i in idx_in_deep])
profile_deep = np.matrix(reviewed_businesses.stars) * deepnn_features_reviewed
print 'Done'

# Given un-reviewed business, compute cosine similarity to user's profile
print '**Computing similarity to all businesses...'
idx_new = range(100) 
#[pd.Index(df_business.business_id).get_loc(b) for b in df_business.business_id if b not in reviewed_businesses.business_id]
features = OHE_union.transform(df_business.ix[idx_new])
similarity = np.asarray(profile * features.T) * 1./(norm(profile) * norm(features, axis = 1))

deep_features_new_bizs = np.array([deepnn_business_features[i] for i in idx_new]) 
similarity_deep = np.asarray(profile_deep * deep_features_new_bizs.T) * 1./(norm(profile_deep) * norm(deep_features_new_bizs, axis = 1))

print 'Done'

deep_coefficient = 1
combined_similarity = np.array([similarity[i] + deep_coefficient * (similarity_deep[i]) for i in range(len(similarity))])

# Output: recommend the most similar business
idx_recommendation = similarity.argmax()
idx_recommendation_deep = combined_similarity.argmax()

print idx_recommendation
print idx_recommendation_deep

print '\n**********'
print 'Hi ' + df_user.name[df_user.user_id == user].iget_value(0) + '!'
print 'We recommend you to visit ' + df_business.name[idx_recommendation] + ' located at '
print df_business.address[idx_recommendation]
print '**********'

## -------------------
## COLLABORATIVE FILTERING
## -------------------
# print '*** Using Collaborative Filtering for Recommendation ***'
# 
# df_review['stars'] = df_review.groupby('business_id')['stars'].transform(lambda x : x - x.mean())
# 
# def get_idx(user_id): 
#     global running_index
#     running_index = running_index + 1
#     return pd.Series(np.zeros(len(user_id)) + running_index) 
# # For speed, get_idx assumes df_review and df_user contain the same users, and is fed in sorted order.
# running_index = -1 
# df_review['user_idx'] = df_review.groupby('user_id')['user_id'].transform(get_idx)
# 
# # Work in terms of sparse matrix
# print '** Processing utility matrix...'
# 
# def convert_to_sparse(group):
#     ratings = coo_matrix( (np.array(group['stars']), (np.array(group['user_idx']), np.zeros(len(group)))), 
#                           shape = (len(df_user), 1) ).tocsc()
#     return ratings / np.sqrt(float(ratings.T.dot(ratings).toarray()))
# 
# utility = df_review.groupby('business_id')[['stars', 'user_idx']].apply(convert_to_sparse) 
# 
# # Get top recommendatiokns
# print '** Generating recommendations...'
# 
# def cosine_similarity(v1, v2):
#     return float(v1.T.dot(v2).toarray())
# 
# def get_recommended_businesses(n, business_id):
#     util_to_match = utility[utility.index == business_id]
#     similarity = utility.apply(lambda x: cosine_similarity(util_to_match.values[0], x))
#     similarity.sort(ascending=False)
#     return similarity[1:(n+1)]
# 
# fav_business = df_review.business_id[ df_review.stars[ df_review.user_id == user ].argmax() ]
# 
# rec = pd.DataFrame(get_recommended_businesses(5, fav_business), columns=['similarity'])
# rec['name'] = [ df_business.name[ df_business.business_id == business_id ].values[0] for business_id in rec.index]
# print 'Done'
# 
# # Output recommendation
# print 'Hi ' + df_user.name[df_user.user_id == user].values[0] + '!\nCheck out these businesses!'
# for name in rec.name:
#     print name