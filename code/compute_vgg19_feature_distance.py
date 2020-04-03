import numpy as np
import argparse
import os

### Function to compute the distance between features on each trial, given a particular model layer.
def compute_feature_distance(obs_lay, obs_rf, trial_params, feature_dir, filenames):
  feature_distance = np.zeros((len(trial_params['image']), len(trial_params['layer']), 
                               len(trial_params['poolsize']), len(trial_params['sample'])+1))
  if obs_rf == 'activ':
      feat = 'feats'
  elif obs_rf == 'diag':
      feat = 'diagonal'
  else:
      feat = '{}gramian'.format(obs_rf)
  features = np.load('{}/VGG19_{}_output_{}.npy'.format(feature_dir, obs_lay, feat))

  for i, img in enumerate(trial_params['image']):
    print('---{}---'.format(img))
    for l, layer in enumerate(trial_params['layer']):
      for p, poolsize in enumerate(trial_params['poolsize']):
        # First get original features
        out = [x for x, v in enumerate(filenames) if 'originals/{}'.format(img) in v]
        assert len(out)==1
        orig_feat = features[out[0],:]
        # Next, get texture features
        idx = [x for x, v in enumerate(filenames) if 'textures/{}_{}_{}'.format(poolsize, layer, img) in v]
        assert len(idx) > 1, 'Error - {} texture sample of {} {} {} were found. We need at least 2'.format(len(idx), poolsize, layer, img)
        smp_feat1 = features[idx[0],:]
        smp_feat2 = features[idx[1],:]
        feature_distance[i,l,p,0] = 1-np.corrcoef(orig_feat, smp_feat1)[0,1]
        feature_distance[i,l,p,1] = 1-np.corrcoef(orig_feat, smp_feat2)[0,1]
        # Last one is comparing the features to each other.
        feature_distance[i,l,p,2] = 1-np.corrcoef(smp_feat1, smp_feat2)[0,1]
  return feature_distance


if __name__ == '__main__':
  # Define directory in which features are found
  feature_dir = '/scratch/groups/jlg/texture_stimuli/color/deepnet_features'

  images = ['face', 'jetplane', 'elephant', 'sand', 'lawn', 'dirt', 'tulips', 'fireworks', 'bananas']
  
  trial_params = {'layer': ['pool1', 'pool2', 'pool4'], 'image': images, 
                  'poolsize': ['1x1', '2x2', '3x3', '4x4'], 'sample':[1,2]}
  observer_params = {'layer': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 
                               'conv3_2', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4', 
                               'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5'], 
                    'poolsize': ['1x1', 'activ', 'diag']}

  filenames = np.load('{}/VGG19_filenames.npy'.format(feature_dir))

  feature_distance = {}
  for obs_rf in observer_params['poolsize']:
    for obs_lay in observer_params['layer']:
      model = '{}_{}'.format(obs_rf, obs_lay)
      print('---Computing feature distances for model layer {} ----'.format(model))

      feature_distance[model] = compute_feature_distance(obs_lay, obs_rf, trial_params, feature_dir, filenames)

  feature_distance['trial_params'] = trial_params
  np.save('/scratch/users/akshayj/texOdd/VGG19_featuredistances.npy', feature_distance)
