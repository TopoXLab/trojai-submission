import torch
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump
from utils.models import load_ground_truth

DATA_PATH = '/scr/TrojAI23/object-detection-feb2023-train'

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


def get_features_and_labels_by_model_class(models_dirpath: str):
    ### A: SSD, B: FasterRCNN, C: DetrForObjectDetection

    model_A_features, model_B_features, model_C_features, model_A_labels, model_B_labels, model_C_labels = [], [], [], [], [], []

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
        class_and_features_by_model = get_model_features(os.path.join(model_path, "model.pt"))
        model_class, model_features = class_and_features_by_model['model_class'], class_and_features_by_model['features']
        model_ground_truth = load_ground_truth(model_path)
        if model_class == 'SSD':
            model_A_features.append(model_features)
            model_A_labels.append(model_ground_truth)
        elif model_class == 'FasterRCNN':
            model_B_features.append(model_features)
            model_B_labels.append(model_ground_truth)
        elif model_class == 'DetrForObjectDetection':
            model_C_features.append(model_features)
            model_C_labels.append(model_ground_truth)
        else:
            print(model_class, " is an unknown model!!!!!!")

    return {'model_A_features': np.asarray(model_A_features), 'model_B_features': np.asarray(model_B_features), 'model_C_features': np.asarray(model_C_features),
            'model_A_labels': np.asarray(model_A_labels), 'model_B_labels': np.asarray(model_B_labels), 'model_C_labels': np.asarray(model_C_labels)}


def get_model_features(model_filepath):
    model = torch.load(model_filepath)
    model_class = model._get_name()
    if model_class == 'DetrForObjectDetection':
        model_backbone = model.model.backbone
    else:
        model_backbone = model.backbone

    #num_of_params = sum(p.numel() for p in model.parameters())/1000.0

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    features = []

    if model_class == 'SSD':
        features = _get_eigen_vals(all_backbone_params, 0, 5)
    elif model_class == 'FasterRCNN':
        features = _get_eigen_vals(all_backbone_params, 0, 5)
    elif model_class == 'DetrForObjectDetection':
        features = _get_eigen_vals(all_backbone_params, 0, 5)
    else:
        print(model_class, " is an unknown model!!!!!!")


    # if num_of_params == 41755.2860:  # model A
    #     model_class = 'A'
    #     features = _get_eigen_vals(all_backbone_params, 0, 5) # 1, 3
    # elif num_of_params == 35641.8260:  # model B
    #     model_class = 'B'
    #     features = _get_eigen_vals(all_backbone_params, 0, 5) # 0, 4

    return {'model_class': model_class, 'features': features}


def _get_eigen_vals(all_backbone_params, idx_low=0, idx_high=3):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) > 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params,False)
                squared_singular_values = singular_values**2
                # top_five_sq_sv = squared_singular_values[:5]
                features += squared_singular_values.tolist()
                num_layers += 1
            num_layers += 1
    return features


def _get_model_filepath(model_num: int) -> os.path:
    num_as_str = str(100000000 + model_num)[1:]
    model_id = f"id-{num_as_str}"
    return os.path.join(DATA_PATH, 'models', model_id, 'model.pt')

if __name__ == "__main__":
    models_dirpath = '/scr/TrojAI23/object-detection-feb2023-train/models'
    model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
    all_features_and_labels = get_features_and_labels_by_model_class(model_path_list)
    model_A_features, model_A_labels = all_features_and_labels['model_A_features'], all_features_and_labels['model_A_labels']
    model_B_features, model_B_labels = all_features_and_labels['model_B_features'], all_features_and_labels['model_B_labels']
    model_C_features, model_C_labels = all_features_and_labels['model_C_features'], all_features_and_labels['model_C_labels']

    np.save('features_A', model_A_features)
    np.save('features_B', model_B_features)
    np.save('features_C', model_C_features)
    np.save('labels_A', model_A_labels)
    np.save('labels_B', model_B_labels)
    np.save('labels_C', model_C_labels)


    base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss',  bootstrap=True, random_state=0)
    clf_A, clf_B, clf_C = CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5)
    clf_A.fit(model_A_features, model_A_labels)
    clf_B.fit(model_B_features, model_B_labels)
    clf_C.fit(model_C_features, model_C_labels)

    dump(clf_A, 'classifier_model_A_05.joblib')
    dump(clf_B, 'classifier_model_B_05.joblib')
    dump(clf_C, 'classifier_model_C_05.joblib')