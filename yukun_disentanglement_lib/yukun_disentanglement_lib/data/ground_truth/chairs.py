# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""chair data sets used for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yukun_disentanglement_lib.data.ground_truth import ground_truth_data
import json
import numpy as np
from yukun_disentanglement_lib.data.ground_truth.utils import get_image


class Chairs(ground_truth_data.GroundTruthData):
  """Dummy image data set of random noise used for testing."""

  def __init__(self):
    self.factor_updated = False
    self.images, self.factors = self._load_data()
    
    #print(z_store.shape)
    #print(len(images))
    #print(images[0].shape)

  @property
  def num_factors(self):
    return 3

  #@property
  #def factors_num_values(self):
  #  return [5] * 10

  @property
  def observation_shape(self):
    return [64, 64, 3]

  def _load_data(self):
    TRAIN_STOP=200000
    #TRAIN_STOP=2000
    data = json.load(open('/hdd_c/data/MITFaces/img_store_chairs'))[:TRAIN_STOP]
    images = [get_image('/hdd_c/data/MITFaces/notebooks/'+name,
                    input_height=128,
                    input_width=128,
                    resize_height=64,
                    resize_width=64,
                    is_crop=False)/255. for name in data]
    print('finish reading chair images')
    images = np.asarray(images)
    print('finish converting chair images to numpy array')
    factors=np.load('../../MITFaces/z_store_new_chairs')[:TRAIN_STOP]
    print('finish reading factors')
    return images, factors

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    #print('sample {} factors'.format(num))
    self.indices = random_state.randint(len(self.images), size=num)
    self.factor_updated = True
    #print(self.factors[self.indices])
    return self.factors[self.indices]

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    #print('sample obs, factor shape: {}'.format(factors.shape))
    if not self.factor_updated:
        raise NotImplementedError
    self.factor_updated = False
    obs = self.images[self.indices]
    #print(obs.shape)
    
    return obs