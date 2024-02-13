import torch
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
import pickle
from functions import normal_distribution, compute_object_detection_jacobian_ssd, compute_object_detection_jacobian_fasterrcnn


DATA_PATH = '/scr/weimin/RoundData/round10'

def get_all_features():
    all_features = []
    num_models = 143
    for model_num in range(num_models):
        model_filepath = _get_model_filepath(model_num)
        all_features.append(get_model_features(model_filepath)['features'])

    X = np.asarray(all_features)
    return X

def get_all_labels():
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    y = metadata['poisoned'].to_numpy()
    return y


def get_features_and_labels_by_model_class():
    model_A_features, model_B_features, model_A_labels, model_B_labels = [], [], [], []
    num_models = 144
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    labels = metadata['poisoned'].to_numpy()
    for model_num in range(num_models):
        model_filepath = _get_model_filepath(model_num)
        class_and_features_by_model = get_model_features(model_filepath)
        model_class, model_features = class_and_features_by_model['model_class'], class_and_features_by_model['features']
        if model_class == 'A':
            model_A_features.append(model_features)
            model_A_labels.append(labels[model_num])
        elif model_class == 'B':
            model_B_features.append(model_features)
            model_B_labels.append(labels[model_num])
    return {'model_A_features': np.asarray(model_A_features), 'model_B_features': np.asarray(model_B_features), 
            'model_A_labels': np.asarray(model_A_labels), 'model_B_labels': np.asarray(model_B_labels)}


def get_model_features(model_filepath):
    model = torch.load(model_filepath)
    ## get the eigen value features
    model_backbone = model.backbone
    num_of_params = sum(p.numel() for p in model.parameters())/1000.0

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    model_class, features = '', []
    if num_of_params == 41755.2860:  # model A, fasterrcnn
        model_class = 'A'
        features = _get_eigen_vals(all_backbone_params, 2,3) # 2, 3
    elif num_of_params == 35641.8260:  # model B, ssd
        model_class = 'B'
        features = _get_eigen_vals(all_backbone_params, 2,5) # 2, 5
    
    fea_jac = _get_jac_vec(model_filepath, model_class)
    fea_com = features + fea_jac # 52800

    return {'model_class': model_class, 'features': fea_com}



def _get_jac_vec(model_filepath, model_architecture):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load or generate normal distribution samples
    distribution_fp = 'normal_distribution.v2.resize.pkl'
    if os.path.exists(distribution_fp):
        print('Load sample distribution.')
        with open(distribution_fp, 'rb') as f:
            gaussian_sample = pickle.load(f)
            f.close()
    else:
        print('Generate sample distribution.')
        ## generate normal distribution samples
        gaussian_sample = normal_distribution(device) #3, 478, 640
        with open(distribution_fp, 'wb') as f:
            pickle.dump(gaussian_sample, f)
            f.close()

    ## resize input
    # gaussian_sample = gaussian_sample[:, :12, :16] # resize to (3, 12, 16) to reduce memory usage -> 52416

    # load the model
    pytorch_model = torch.load(model_filepath)
    pytorch_model.to(device)
    pytorch_model.eval()
    torch.backends.cudnn.enabled = False # allow backwards when model.eval()

    # gaussian_sample.retains_grad = True
    # iterate all rand samples
    images = [gaussian_sample]
    output = pytorch_model(images)  #head_outputs_cls_logits - torch.Size([1, 8732, 91]) or [372, 91]
    # output = outputs[1] #head_outputs_cls_logits - torch.Size([1, 8732, 91])
    if model_architecture == 'B':
        jacobian_mat = compute_object_detection_jacobian_ssd(images, output) #  torch.Size([91, 3, 478, 640])
    else:
        jacobian_mat = compute_object_detection_jacobian_fasterrcnn(images, output) #  torch.Size([91, 3, 478, 640])

    jacobian_mean = jacobian_mat.view(1, -1) # (1, ..) # 83516160
    jacobian_mean = jacobian_mean.cpu().detach().numpy()[0]
    # print('samples_jacobian ([83516160]])', jacobian_mean.shape)
    jacobian_mean = jacobian_mean.tolist()

    return jacobian_mean


def _get_eigen_vals(all_backbone_params, idx_low=0, idx_high=3):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) > 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[0], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params,False)
                squared_singular_values = singular_values**2
                features += squared_singular_values.tolist()
        num_layers += 1
    return features


def _get_model_filepath(model_num: int) -> os.path:
    num_as_str = str(100000000 + model_num)[1:]
    model_id = f"id-{num_as_str}"
    return os.path.join(DATA_PATH, 'round10-train-dataset', model_id, 'model.pt')

if __name__ == "__main__":
    all_features_and_labels = get_features_and_labels_by_model_class()
    model_A_features, model_B_features, model_A_labels, model_B_labels = all_features_and_labels['model_A_features'], all_features_and_labels['model_B_features'], all_features_and_labels['model_A_labels'], all_features_and_labels['model_B_labels']
    clf_A = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.0005, max_depth=3, subsample=0.7, max_features='sqrt', random_state=0, loss='log_loss')
    clf_B = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.00007, max_depth=3, subsample=0.7, max_features='sqrt', random_state=0, loss='log_loss')
    clf_A.fit(model_A_features, model_A_labels)
    clf_B.fit(model_B_features, model_B_labels)
    dump(clf_A, 'classifier_model_A_23_jac.joblib')
    dump(clf_B, 'classifier_model_B_25_jac.joblib')


