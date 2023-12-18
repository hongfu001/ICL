import  numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

import random

from algorithms.sampling.utils import create_sample_alg
from algorithms.sampling.base_sampling import Sampling
from common.utils import calculate_sentence_transformer_embedding


class kCenterGreedy(Sampling):

  def __init__(self, X, metric='euclidean'):
    self.X = X
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []


  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, features, already_selected, N):
  
    try:
      print('Getting transformed features...')
      self.features = features
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    if already_selected is None:
        already_selected = []
    self.already_selected = already_selected
    print(self.already_selected)

    new_batch = []

    for _ in range(N):
      if self.already_selected == []:
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      assert ind not in already_selected
      
      
      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      
      if self.already_selected is None:
          self.already_selected = []
      else:
          self.already_selected.append(ind)
    
    return self.already_selected
  
  def demo_selecting(self, dataset, sampler):
    dataset_text_embedding=np.array(calculate_sentence_transformer_embedding(dataset['text'].tolist()))
    k_center_select_id = self.select_batch_( dataset_text_embedding, already_selected= None, N=len(sampler))
    sampler = dataset[dataset['id'].isin(k_center_select_id)]
    alg_sample_obj = create_sample_alg(self.args, 'random')
    
    concat_dataset = pd.DataFrame()
    for epoch in range(1,10):
      np.random.seed(epoch)
      random.seed(epoch)
      dataset_one = alg_sample_obj.sample_demo(dataset, sampler)
      concat_dataset = pd.concat([concat_dataset, dataset_one])
      
    concat_dataset_text_embedding=np.array(calculate_sentence_transformer_embedding(concat_dataset['text'].tolist()))
    k_center_select_id = self.select_batch_(concat_dataset_text_embedding, already_selected= None, N=len(sampler))
    
    dataset_final = dataset[concat_dataset['id'].isin(k_center_select_id)]
    return dataset_final