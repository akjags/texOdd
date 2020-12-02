import os, pickle, tqdm, glob
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from PIL import Image

np.random.seed(0)
torch.manual_seed(0)

# Function to get pytorch models
def get_model(model_name='resnet', layer='conv'):
  if model_name == 'resnet':
    assert layer in ['conv', 'avgpool', 'readout'], 'layer must be: conv, avgpool, or readout'
    resnet18 = models.resnet18(pretrained=True)
    if layer == 'conv':
      model = nn.Sequential(*list(resnet18.children())[:-2])
    elif layer == 'avgpool': 
      model = nn.Sequential(*list(resnet18.children())[:-1])
    else:
      model = resnet18

  elif model_name == 'alexnet':
    assert layer in ['conv', 'readout'], 'layer must be one of: conv, readout'
    alexnet = models.alexnet(pretrained=True)
    if layer == 'conv':
      #model = nn.Sequential(*list(alexnet.children())[-1])
      model = list(alexnet.children())[0] # Exclude FC layers
    else:
      model = alexnet
  else
    print('Sorry no other models have been implemented yet')
  return model

# loader function to load the images
def load_image(filepath):
  if os.path.isfile(filepath+'.png'):
    filepath = filepath + '.png'
  elif os.path.isfile(filepath+'.jpg'):
    filepath = filepath + '.jpg'
  else:
    assert os.path.isfile(filepath)
 
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((imsize,imsize)),
                torchvision.transforms.ToTensor(),
                normalize])
  im = Image.open(filepath).convert('RGB')
  im = transform(im)
  return im

def get_features(model, filepath):
  img = load_image(filepath)
  return model[img]

def distance(vec1, vec2, dist_metric = 'correlation'):
  assert dist_metric in ['correlation']
  if dist_metric == 'correlation':
    return 1-np.corrcoef(vec1, vec2)[0,1]


# For each image class, load images, extract features, and compute distance.
def compute_feature_distance(model, trial_params, orig_dir, tex_dir):
  feature_distance = np.zeros((len(trial_params['image']), len(trial_params['layer']), 
                               len(trial_params['poolsize']), len(trial_params['sample'])+1))

  for i, img in tqdm(enumerate(trial_params['image']), total=len(trial_params['image'])):
    for l, layer in enumerate(trial_params['layer']):
      for p, poolsize in enumerate(trial_params['poolsize']): 
        orig_feat = get_features(model, '{}_{}'.format(orig_dir, img))
        tex_feat1 = get_features(model, '{}/{}_{}_{}_smp1'.format(tex_dir, poolsize, layer, img))
        tex_feat2 = get_features(model, '{}/{}_{}_{}_smp2'.format(tex_dir, poolsize, layer, img))

        # Compute the distance between each pair of images (orig<-->s1<-->s2)
        feature_distance[i,l,p,0] = distance(orig_feat, tex_feat1)
        feature_distance[i,l,p,1] = distance(orig_feat, tex_feat2)
        # Last one is comparing the synthesized samples to each other.
        feature_distance[i,l,p,2] = distance(tex_feat1, tex_feat2)


if __name__ == '__main__':
  # Define directory in which features are found
  orig_dir = '/scratch/groups/jlg/texture_stimuli/color/originals'
  tex_dir = '/scratch/groups/jlg/texture_stimuli/color/textures'

  images = ['face', 'jetplane', 'elephant', 'sand', 'lawn', 'tulips', 'dirt', 
            'fireworks', 'bananas', 'apple', 'bear', 'cat', 'cruiseship', 
            'dalmatian', 'ferrari', 'greatdane', 'helicopter', 'horse', 
            'house', 'iphone', 'laptop', 'quarterback', 'samosa', 'shoes', 
            'stephcurry', 'tiger', 'truck', 'leaves', 'stars', 'tiles', 
            'worms', 'bumpy', 'spiky', 'clouds', 'crowd', 'forest', 'frills']
  trial_params = {'layer': ['pool1', 'pool2', 'pool4'], 'image': images, 
                  'poolsize': ['1x1', '2x2', '3x3', '4x4'], 'sample':[1,2]}

  observer_params = {'layer': ['conv', 'avgpool', 'readout'],
                    'poolsize': ['activ'],
                    'model_name': 'resnet',
                    'dist_metric': 'correlation'}

  feature_distance = {}
  for obs_rf in observer_params['poolsize']:
    for obs_lay in observer_params['layer']:
      obs_model = '{}_{}'.format(obs_rf, obs_lay)
      model = get_model('resnet', obs_lay)

      feature_distance[obs_model] = compute_feature_distance(model, trial_params, orig_dir, tex_dir)

  feature_distance['trial_params'] = trial_params
  feature_distance['observer_params'] = observer_params
  np.save('/scratch/users/akshayj/texOdd/{}_featuredistance_{}.npy'.format(observer_params['model_name'], observer_params['dist_metric']), feature_distance)


