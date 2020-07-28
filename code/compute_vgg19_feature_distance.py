import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave
from tqdm import tqdm
import sys, pickle, argparse, os
# Add folder containing texture synthesis and VGG19 model files.
sys.path.insert(1, '/home/users/akshayj/TextureSynthesis/tensorflow_synthesis')
from VGG19 import *
from TextureSynthesis import TextureSynthesis

### Function to compute the distance between features on each trial, given a particular model layer.
def compute_feature_distance(vgg19, obs_lay, obs_rf, trial_params, orig_dir, texture_dir):
  '''
  Function to compute the distance between features on each trial, given a particular model layer.
  '''

  feature_distance = np.zeros((len(trial_params['image']), len(trial_params['layer']), 
                               len(trial_params['poolsize']), len(trial_params['sample'])+1))

  # NEW
  ts = TextureSynthesis(vgg19, np.zeros((1,256,256,3)), {obs_lay: 1e9}, 1 if obs_rf in ['activ', 'diag'] else int(obs_rf[0]))

  for i, img in tqdm(enumerate(trial_params['image']), total=len(trial_params['image'])):
    #print('---{}---'.format(img))
    for l, layer in enumerate(trial_params['layer']):
      for p, poolsize in enumerate(trial_params['poolsize']):
        # Get feature vector at this observer layer for original image and texture samples
        orig_feat = get_features(ts, '{}/{}'.format(orig_dir, img), obs_rf, obs_lay)
        tex_feat1 = get_features(ts, '{}/{}_{}_{}_smp1'.format(texture_dir, poolsize, layer, img), obs_rf, obs_lay)
        tex_feat2 = get_features(ts, '{}/{}_{}_{}_smp2'.format(texture_dir, poolsize, layer, img), obs_rf, obs_lay)

        # Compute the distance between each pair of features (orig<-->sample1<-->sample2)
        feature_distance[i,l,p,0] = distance(orig_feat, tex_feat1)
        feature_distance[i,l,p,1] = distance(orig_feat, tex_feat2)
        # Last one is comparing the features to each other.
        feature_distance[i,l,p,2] = distance(tex_feat1, tex_feat2)

  del ts
  return feature_distance


def distance(vec1, vec2, dist_metric = 'correlation'):

  if dist_metric == 'correlation':
    return 1-np.corrcoef(vec1, vec2)[0,1]


def get_features(ts, filepath, obs_rf, obs_lay):
  image = preprocess(filepath)

  if obs_rf == 'activ':
    feat = ts._get_activations(original_image = image)[obs_lay]
  elif obs_rf == 'diag':
    feat = np.diag(ts._get_gramian(original_image = image)[obs_lay].squeeze())
  else:
    feat = ts._get_gramian(original_image = image)[obs_lay]
  return feat.ravel()

def preprocess(path):
    '''
    Reads in an image and preprocesses it for input to VGG19.

    Arguments:
      - path --> a string containing the full path to an image:
        e.g. path='/scratch/groups/jlg/texture_stimuli/color/originals/rocks.png'
          - If you leave off the extension, it will check for either PNG or JPG.
    '''
    if os.path.isfile(path+'.png'):
      path = path + '.png'
    elif os.path.isfile(path+'.jpg'):
      path = path + '.jpg'
    else:
      assert os.path.isfile(path)
      
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    image = imread(path)

    if image.shape[1]!=256 or image.shape[0]!=256:
        image = resize(image, (256,256))

    # add an extra dimension for batch size of 1.
    image = np.reshape(image, ((1,) + image.shape))
    if len(image.shape)<4:
        image = np.stack((image,image,image),axis=3)

    # If there is a Alpha channel, just scrap it
    if image.shape[3] == 4:
        image = image[:,:,:,:3]

    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES

    return image


if __name__ == '__main__':
  # Define directory in which features are found
  orig_dir = '/scratch/groups/jlg/texture_stimuli/color/originals'
  texture_dir = '/scratch/groups/jlg/texture_stimuli/color/textures'

  #images = ['face', 'jetplane', 'elephant', 'sand', 'lawn', 'dirt', 'tulips', 'fireworks', 'bananas']
  images = ['face', 'jetplane', 'elephant', 'sand', 'lawn', 'tulips', 'dirt', 'fireworks', 'bananas', 'apple', 'bear', 'cat', 'cruiseship', 'dalmatian', 'ferrari', 'greatdane', 'helicopter', 'horse', 'house', 'iphone', 'laptop', 'quarterback', 'samosa', 'shoes', 'stephcurry', 'tiger', 'truck', 'leaves', 'stars', 'tiles', 'worms', 'bumpy', 'spiky', 'clouds', 'crowd', 'forest', 'frills']
  trial_params = {'layer': ['pool1', 'pool2', 'pool4'], 'image': images, 
                  'poolsize': ['1x1', '2x2', '3x3', '4x4'], 'sample':[1,2]}
  observer_params = {'layer': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 
                               'conv3_2', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4', 
                               'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5'], 
                    'poolsize': ['1x1', 'activ', 'diag']}

  # Load VGG-19 weights and build model
  with open('/home/users/akshayj/TextureSynthesis/tensorflow_synthesis/vgg19_normalized.pkl', 'rb') as f:
      vgg_weights = pickle.load(f)['param values']

  feature_distance = {}
  for obs_rf in observer_params['poolsize']:
    for obs_lay in observer_params['layer']:
      model = '{}_{}'.format(obs_rf, obs_lay)
      print('---Computing feature distances for model layer {} ----'.format(model))
      tf.reset_default_graph()
      vgg19 = VGG19(vgg_weights)
      vgg19.build_model()

      feature_distance[model] = compute_feature_distance(vgg19, obs_lay, obs_rf, trial_params, orig_dir, texture_dir)

  feature_distance['trial_params'] = trial_params
  feature_distance['observer_params'] = observer_params
  np.save('/scratch/users/akshayj/texOdd/VGG19_featuredistance_correlation.npy', feature_distance)
