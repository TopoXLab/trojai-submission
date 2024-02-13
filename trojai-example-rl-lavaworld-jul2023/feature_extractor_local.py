import torch
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump
from utils.models import load_ground_truth
import argparse
import random
from sklearn.metrics import roc_auc_score

DATA_PATH = '/scr/TrojAI23/rl-lavaworld-jul2023/training'


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--low_layer', type=int)
    parser.add_argument('--high_layer', type=int)
    return parser


def get_all_features():
    all_features = []
    num_models = 144
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


def get_features_and_labels_by_model_class(args, models_dirpath: str):
    ### A: BasicFCModel, B: SimplifiedRLStarter

    model_A_features, model_B_features, model_C_features, model_A_labels, model_B_labels, model_C_labels = [], [], [], [], [], []
    # metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    # metadata = pd.read_csv(metadata_filepath)
    # labels = metadata.ground_truth.map({'clean': 0.0, 'triggered': 1.0}).to_numpy()
    # num_models = labels.shape[0]
    ## check models and parameters
    # models = []
    # models_params = []
    # for model_num in range(num_models):
    #     model_filepath = _get_model_filepath(model_num)
    #     model = torch.load(model_filepath)
    #     #model_backbone = model.backbone
    #     num_of_params = sum(p.numel() for p in model.parameters()) / 1000.0
    #     models.append(model)
    #     models_params.append(num_of_params)

    for model_path in models_dirpath:
        if not os.path.isdir(model_path):
            continue
        class_and_features_by_model = get_model_features(args, os.path.join(model_path, "model.pt"))
        model_class, model_features = class_and_features_by_model['model_class'], class_and_features_by_model['features']
        model_ground_truth = load_ground_truth(model_path)

        if model_class == 'BasicFCModel':
            model_A_features.append(model_features)
            model_A_labels.append(model_ground_truth)
        elif model_class == 'SimplifiedRLStarter':
            model_B_features.append(model_features)
            model_B_labels.append(model_ground_truth)
        else:
            print(model_class, " is an unknown model!!!!!!")

    return {'model_A_features': np.asarray(model_A_features), 'model_B_features': np.asarray(model_B_features),
            'model_A_labels': np.asarray(model_A_labels), 'model_B_labels': np.asarray(model_B_labels)}


def get_model_features(args, model_filepath):
    model = torch.load(model_filepath)
    model_class = model._get_name()
    model_backbone = model

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    features = []

    if model_class == 'BasicFCModel':
        features = _get_eigen_vals_fc(all_backbone_params, args.low_layer, args.high_layer)
    elif model_class == 'SimplifiedRLStarter':
        features = _get_eigen_vals(all_backbone_params, args.low_layer, args.high_layer)
    else:
        print(model_class, " is an unknown model!!!!!!")

    return {'model_class': model_class, 'features': features}


def _get_eigen_vals(all_backbone_params, idx_low=0, idx_high=3):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) > 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values**2
                # top_five_sq_sv = squared_singular_values[:5]
                features += squared_singular_values.tolist()
                # num_layers += 1
            num_layers += 1
        if len(backbone_params.shape) == 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values**2
                # top_five_sq_sv = squared_singular_values[:5]
                features += squared_singular_values.tolist()
                # num_layers += 1
            num_layers += 1
    print(num_layers)
    return features


def _get_eigen_vals_fc(all_backbone_params, idx_low=0, idx_high=3):
    print(idx_low)
    print(idx_high)
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) == 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values**2
                # top_five_sq_sv = squared_singular_values[:5]
                features += squared_singular_values.tolist()
                # num_layers += 1
            num_layers += 1
    return features


def _get_model_filepath(model_num: int) -> os.path:
    num_as_str = str(100000000 + model_num)[1:]
    model_id = f"id-{num_as_str}"
    return os.path.join(DATA_PATH, 'models', model_id, 'model.pt')


if __name__ == "__main__":
    np.random.seed(0)
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    models_dirpath = '/scr/TrojAI23/rl-lavaworld-jul2023/training/models'
    model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
    models_dirpath_leftovers = '/scr/TrojAI23/rl-lavaworld-jul2023/leftovers/models'
    model_path_list_leftovers = sorted([os.path.join(models_dirpath_leftovers, model) for model in os.listdir(models_dirpath_leftovers)])
    model_path_list.extend(model_path_list_leftovers)
    all_features_and_labels = get_features_and_labels_by_model_class(args, model_path_list)
    model_A_features, model_A_labels = all_features_and_labels['model_A_features'], all_features_and_labels['model_A_labels']
    model_B_features, model_B_labels = all_features_and_labels['model_B_features'], all_features_and_labels['model_B_labels']

    np.save('features_A', model_A_features)
    np.save('features_B', model_B_features)
    np.save('labels_A', model_A_labels)
    np.save('labels_B', model_B_labels)
    shuffle_index_A = np.arange(model_A_features.shape[0])
    np.random.shuffle(shuffle_index_A)
    np.random.seed(0)
    shuffle_index_B = np.arange(model_B_features.shape[0])
    np.random.shuffle(shuffle_index_B)

    train_index_A = sorted(shuffle_index_A[:int(0.8 * model_A_features.shape[0])])
    print(train_index_A)
    val_index_A = sorted(shuffle_index_A[int(0.8 * model_A_features.shape[0]):])
    print(val_index_A)
    train_index_B = sorted(shuffle_index_B[:int(0.8 * model_B_features.shape[0])])
    print(train_index_B)
    val_index_B = sorted(shuffle_index_B[int(0.8 * model_B_features.shape[0]):])
    print(val_index_B)

    train_model_A_features = model_A_features[train_index_A, :]
    train_model_A_labels = model_A_labels[train_index_A]
    val_model_A_features = model_A_features[val_index_A, :]
    val_model_A_labels = model_A_labels[val_index_A]
    train_model_B_features = model_B_features[train_index_B, :]
    train_model_B_labels = model_B_labels[train_index_B]
    val_model_B_features = model_B_features[val_index_B, :]
    val_model_B_labels = model_B_labels[val_index_B]

    base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss',  bootstrap=True, random_state=0)
    clf_A, clf_B, clf_C = CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5)
    # clf_A.fit(model_A_features, model_A_labels)
    clf_A.fit(train_model_A_features, train_model_A_labels)
    clf_B.fit(model_B_features, model_B_labels)
    # clf_B.fit(train_model_B_features, train_model_B_labels)

    # dump(clf_A, 'classifier_model_A_'+str(args.low_layer)+str(args.high_layer)+'.joblib')
    dump(clf_B, 'classifier_model_B_'+str(args.low_layer)+str(args.high_layer)+'.joblib')

    #### validation
    probabilities = clf_A.predict_proba(val_model_A_features)
    auc_score_A = roc_auc_score(val_model_A_labels, probabilities[:, 1])
    print(auc_score_A)
    probabilities = clf_B.predict_proba(val_model_B_features)
    auc_score_B = roc_auc_score(val_model_B_labels, probabilities[:, 1])
    print(auc_score_B)



