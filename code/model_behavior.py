from scipy.io import loadmat
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

### Function to load behavioral data from a .mat file
def load_behavioral_data(filepath):
    bd = loadmat(filepath)
    fix_str_vec = lambda vec: np.array([str(vec[0][i][0]) for i in range(len(vec[0]))], dtype='string_')
    for field in ['imNames', 'poolSizes', 'layerNames']:
        bd[field] = fix_str_vec(bd[field])
    for field in ['layer', 'imgFam', 'oddball_layer', 'oddball_poolsize', 'standard_layer', 'standard_poolsize', 'correct', 'accByRuns', 'nTrials', 'nValTrials']:
        bd[field] = np.squeeze(bd[field])
    for field in ['__globals__', '__header__', '__version__', 'dead', 'detected', 'layer']:
        bd.pop(field, None)
    return bd


### Function to compute the distance between features on each trial, given a particular model layer.
def compute_feature_distance(obs_lay, obs_rf, bd, feature_dir):
    feature_distance = np.zeros((bd['nTrials'],))
    trial_feat_dist = np.zeros((3, bd['nTrials']))
    for i in range(bd['nTrials']):
        imgName = bd['imNames'][bd['imgFam'][i]-1]
        
        # First load oddball features
        oLay = bd['oddball_layer'][i]
        if oLay == 0: # Original
            odd_feat = np.load('{}/gram{}_{}.npy'.format(feature_dir, obs_rf, imgName)).item()[obs_lay]
        else:
            if oLay == 1: # P-S
                oPoolName = '1x1'
            else:
                oPoolName = bd['poolSizes'][bd['oddball_poolsize'][i]-1]
            oLayName = bd['layerNames'][bd['oddball_layer'][i]-1]
            odd_feat = np.load('{}/gram{}_{}_{}_{}_smp1.npy'.format(feature_dir, obs_rf, oPoolName, oLayName, imgName)).item()[obs_lay]

        # Then load distractor features
        stdLayName = bd['layerNames'][bd['standard_layer'][i]-1]
        stdPoolName = bd['poolSizes'][bd['standard_poolsize'][i]-1]
        
        if stdLayName == 'PS':
            stdPoolName = '1x1'
        std_feat1 = np.load('{}/gram{}_{}_{}_{}_smp1.npy'.format(feature_dir, obs_rf, stdPoolName, stdLayName, imgName)).item()[obs_lay]
        std_feat2 = np.load('{}/gram{}_{}_{}_{}_smp2.npy'.format(feature_dir, obs_rf, stdPoolName, stdLayName, imgName)).item()[obs_lay]

        # Compute distance between oddball and each of the two distractors.
        trial_featDist = np.array([1-np.corrcoef(odd_feat.ravel(), std_feat1.ravel())[0,1],
                          1-np.corrcoef(odd_feat.ravel(), std_feat2.ravel())[0,1],
                          1-np.corrcoef(std_feat1.ravel(), std_feat2.ravel())[0,1]])

        trial_feat_dist[:,i] = trial_featDist
        # Calculate how far each image is, on average, from the other two images.
        #image_separation = [np.mean(trial_featDist[0:-1]), np.mean(trial_featDist[[0,2]]), np.mean(trial_featDist[[1:]])]
        feature_distance[i] = np.mean(trial_featDist[:2])

        if i % 1000 == 0:
            print i, feature_distance[i]
    return trial_feat_dist
        
### Function to compute the log loss between the true labels and predicted values.
def get_log_loss(yt, yp):
    return (yt*np.log(yp) + (1 - yt)*np.log(1 - yp))

### Function to fit the logistic regression to predict probability of getting a trial correct, given feature distances.
def fit_logistic_regression(bd, obs_lays, obs_rfs, feature_distance):
    valid = np.logical_not(np.isnan(bd['correct']))
    loss = np.zeros((len(obs_lays),len(obs_rfs)))
    cv_preds = np.zeros((len(obs_lays), len(obs_rfs), np.sum(valid)))
    for i, obs_lay in enumerate(obs_lays):
        for j, obs_rf in enumerate(obs_rfs):
            feat_dist = feature_distance['{}_{}'.format(obs_rf, obs_lay)]

            # get cross validated predictions.
            logReg = LogisticRegression(class_weight='balanced')
            y_pred = cross_val_predict(logReg, feat_dist[valid].reshape(-1,1), bd['correct'][valid], cv=10, method='predict_proba')
            loss[i,j] = -np.sum(get_log_loss(bd['correct'][valid], y_pred[:,1]))
            cv_preds[i,j,:] = y_pred[:,1]

            print('{0} {1} Model: LogLoss={2:.3f}'.format(obs_rf, obs_lay, loss[i,j]))
    return loss, cv_preds, bd['correct'][valid]

##### Function to compute softmax
def softmax(X, axis=None):
    if axis is None:
        return np.divide(np.exp(X), np.sum(np.exp(x)))
    else:
        return np.divide(np.exp(X), np.sum(np.exp(X), axis=axis))

##### Function to transform the feature distances into choice probabilities using a softmax
def compute_softmax_choice_probability(feature_distance):
    # 1. Convert pairwise distances into mean separation for each image.
    image_distance = np.zeros(feature_distance.shape)
    image_distance[0,:] = np.mean(feature_distance[[0,1],:], axis=0) # Oddity
    image_distance[1,:] = np.mean(feature_distance[[0,2],:], axis=0) # Non-odd #1
    image_distance[2,:] = np.mean(feature_distance[[1,2],:], axis=0) # Non-odd #2

    # 2. Pass through a Gaussian kernel.
    
    # 3. Softmax to get each choice probability.
    softmax_choice_probs = softmax(image_distance, axis=0)
    model_choices = np.argmax(softmax_choice_probs, axis=0)

    return softmax_choice_probs, model_choices


if __name__ == "__main__":
    # 1. Load behavioral data
    subj = 's097'
    wd = '/home/users/akshayj/texOdd'
    bd = load_behavioral_data('{}/data/{}_behavior.mat'.format(wd, subj))

    # 2. Compute feature distance
    feature_dir = '/scratch/groups/jlg/gram_texOB'
    obs_lays = ['conv1_1', 'pool1', 'pool2', 'pool4']
    obs_rfs = ['1x1', '2x2', '3x3', '4x4', '5x5', '6x6']

    save_path = '{}/data/{}_feature_dist.npy'.format(wd, subj)
    if os.path.isfile(save_path):
        print('Loading precomputed feature distances')
        feature_distance = np.load(save_path).item()
    else:
        feature_distance = {}

    for obs_rf in obs_rfs:
        for obs_lay in obs_lays:
            key = '{}_{}'.format(obs_rf, obs_lay)
            if key not in feature_distance:
                print('--- {} ---'.format(key))
                feature_distance[key] = compute_feature_distance(obs_lay, obs_rf, bd, feature_dir)
                np.save('{}/data/{}_feature_dist.npy'.format(wd, subj), feature_distance)
                print('Saving to {}/data/{}_feature_dist.npy'.format(wd, subj))
            else:
                print('{} already computed; skipping...'.format(key)) 

    # 3. Fit logistic regression
    results = {'obs_lays': obs_lays, 'obs_rfs': obs_rfs}
    for obs_rf in obs_rfs:
        for obs_lay in obs_lays:
            key = '{}_{}'.format(obs_rf, obs_lay)
            softmax_choice_prob, model_choice = compute_softmax_choice_probability(feature_distance[key])
            results[key] = {'softmax_choice_prob':softmax_choice_prob, 'model_choice': model_choice}
    np.save('{}/results/{}_softmax_model.npy'.format(wd, subj), results)
    print('Saved softmax results to {}/results'.format(wd))
    #results['loss'], results['predictions'], results['responses'] = fit_logistic_regression(bd, obs_lays, obs_rfs, feature_distance)
    #np.save('{}/results/{}_prediction_results.npy'.format(wd, subj), results)


