from scipy.io import loadmat


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
def compute_feature_distance(obs_lay, bd, feature_dir):
    feature_distance = np.zeros((bd['nTrials'],))
    trial_correct = np.zeros((bd['nTrials'],))
    for i in range(bd['nTrials']):
        imgName = bd['imNames'][bd['imgFam'][i]-1]
        
        # First load oddball features
        oLay = bd['oddball_layer'][i]
        if oLay == 0: # Original
            odd_feat = np.load('{}/gram_{}.npy'.format(feature_dir, imgName)).item()[obs_lay]
        else:
            if oLay == 1: # P-S
                oPoolName = '1x1'
            else:
                oPoolName = bd['poolSizes'][bd['oddball_poolsize'][i]-1]
            oLayName = bd['layerNames'][bd['oddball_layer'][i]-1]
            odd_feat = np.load('{}/gram_{}_{}_{}_smp1.npy'.format(feature_dir, oPoolName, oLayName, imgName)).item()[obs_lay]

        # Then load distractor features
        stdLayName = bd['layerNames'][bd['standard_layer'][i]-1]
        stdPoolName = bd['poolSizes'][bd['standard_poolsize'][i]-1]
        
        if stdLayName == 'PS':
            stdPoolName = '1x1'
        std_feat1 = np.load('{}/gram_{}_{}_{}_smp1.npy'.format(feature_dir, stdPoolName, stdLayName, imgName)).item()[obs_lay]
        std_feat2 = np.load('{}/gram_{}_{}_{}_smp2.npy'.format(feature_dir, stdPoolName, stdLayName, imgName)).item()[obs_lay]

        # Compute distance between oddball and each of the two distractors.
        trial_featDist = np.array([1-np.corrcoef(odd_feat.ravel(), std_feat1.ravel())[0,1],
                          1-np.corrcoef(odd_feat.ravel(), std_feat2.ravel())[0,1],
                          1-np.corrcoef(std_feat1.ravel(), std_feat2.ravel())[0,1]])
        feature_distance[i] = np.mean(trial_featDist[:2])
        
        if feature_distance[i] > np.mean(trial_featDist[[1,2]]) and feature_distance[i] > np.mean(trial_featDist[[0,2]]):
            trial_correct[i] = 1
        
        if i % 250 == 0:
            print i, feature_distance[i]
    return feature_distance, trial_correct
        
### Function to compute the log loss between the true labels and predicted values.
def get_log_loss(yt, yp):
    return (yt*np.log(yp) + (1 - yt)*np.log(1 - yp))

### Function to fit the logistic regression to predict probability of getting a trial correct, given feature distances.
def fit_logistic_regression(bd, obs_lays, feature_distance):
    valid = np.logical_not(np.isnan(bd['correct']))
    loss = np.zeros((len(obs_lays),))
    for i, obs_lay in enumerate(obs_lays):
        feat_dist = feature_distance[obs_lay]

        # get cross validated predictions.
        logReg = LogisticRegression(class_weight='balanced')
        y_pred = cross_val_predict(logReg, feat_dist[valid].reshape(-1,1), bd['correct'][valid], cv=10, method='predict_proba')
        loss[i] = -np.sum(get_log_loss(bd['correct'][valid], y_pred[:,1]))
    return loss
