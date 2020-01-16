import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filepath = '/Users/akshay/proj/jgl/data/texOdd';
sessions = os.listdir(filepath)

idx = 0
all_data = []
arrvel = lambda x: np.array(x).ravel()

# LOAD DATA FROM FILE
for session in sessions:
	files = os.listdir('{}/{}'.format(filepath, session))
	correct, layer, image, poolsize = [], [], [], []
	for file in files:
		#print(file)
		with open('{}/{}/{}'.format(filepath, session, file), 'r') as json_file:
			data = json.load(json_file)

			if 'correct' in data and len(data['correct'])==20:
				correct.append(data['correct'])
				layer.append(data['layer_name'])
				image.append(data['img_name'])
				poolsize.append(data['poolsize'])

				#print('NumTrials: {}, Accuracy: {}'.format(len(correct), np.nanmean(correct)))
	
	correct, layer = np.array(correct, dtype=np.float).ravel(), arrvel(layer)
	image, poolsize = arrvel(image), arrvel(poolsize)
	if len(correct) > 0:
		subj_data = {'correct': correct, 'layer': layer, 'image': image, 'poolsize': poolsize}
		all_data.append(subj_data)

pools = ['1x1', '2x2', '4x4']
layers = ['pool1', 'pool2', 'pool4']
correct, layer, image, poolsize = [], [], [], []
for si, subj in enumerate(all_data):
	print('---Subject {} ---'.format(si+1))
	print('Mean accuracy: {}'.format(np.nanmean(subj['correct'])))

	lay_acc, pool_acc = np.zeros((len(layers),)), np.zeros((len(pools),))
	N_lays, N_pools = np.zeros((len(layers),)), np.zeros((len(pools),))
	for li, lay in enumerate(layers):
		lay_acc[li] = np.nanmean(subj['correct'][subj['layer']==lay])
		N_lays[li] = np.sum(~np.isnan(subj['correct'][subj['layer']==lay]))
	print(lay_acc, N_lays)

	for pi, pool in enumerate(pools):
		pool_acc[pi] = np.nanmean(subj['correct'][subj['poolsize']==pool])
		N_pools[pi] = np.sum(~np.isnan(subj['correct'][subj['poolsize']==pool]))
	print(pool_acc, N_pools)

	# Combine data across subjects
	correct.append(subj['correct'])
	layer.append(subj['layer'])
	image.append(subj['image'])
	poolsize.append(subj['poolsize'])
combined = dict()
combined['correct'], combined['layer'], combined['image'], combined['poolsize'] = arrvel(correct), arrvel(layer), arrvel(image), arrvel(poolsize)
np.save('/Users/akshay/proj/texOdd/data/mTurk_behavior.npy', combined)
print('Saving behavioral data to ~/proj/texOdd/data/mTurk_behavior.npy')

def plot_combined_results(combined):
	# Plot results combined across subjects.
	lay_acc, pool_acc = np.zeros((len(layers),)), np.zeros((len(pools),))

	fig = plt.figure(figsize=(10,5));
	for li, lay in enumerate(layers):
		lay_acc[li] = np.nanmean(combined['correct'][combined['layer']==lay])
	ax = plt.subplot(2,1,1);
	plt.plot(np.arange(len(layers)), lay_acc, 'o')
	#plt.ylim([.2, 1]);
	sns.despine()

	for pi, pool in enumerate(pools):
		pool_acc[pi] = np.nanmean(combined['correct'][combined['poolsize']==pool])
	ax = plt.subplot(2,1,2);
	plt.plot(np.arange(len(pools)), pool_acc, 'o')
	#plt.ylim([.2, 1]);
	sns.despine()
	plt.show()

def plot_image_results(combined):
	# Plot results split by image
	images = np.unique(image)
	fig = plt.figure(figsize=(15,5));

	for ii, img in enumerate(images):
		lay_acc, pool_acc = np.zeros((len(layers),)), np.zeros((len(pools),))

		for li, lay in enumerate(layers):
			lay_acc[li] = np.nanmean(combined['correct'][np.logical_and(combined['image']==img, combined['layer']==lay)])
		ax = plt.subplot(2,len(images),ii+1);
		plt.plot(np.arange(len(layers)), lay_acc, 'o');
		plt.ylim([.2, 1]);
		plt.title(img)
		ax.set_xlabel('Layer');

		for pi, pool in enumerate(pools):
			pool_acc[pi] = np.nanmean(combined['correct'][np.logical_and(combined['image']==img, combined['poolsize']==pool)])
		ax = plt.subplot(2,len(images),ii+1+len(images));
		plt.plot(np.arange(len(pools)), pool_acc, 'o')
		plt.ylim([.2, 1]);
		ax.set_xlabel('Pool Size')
		sns.despine()
	plt.show()