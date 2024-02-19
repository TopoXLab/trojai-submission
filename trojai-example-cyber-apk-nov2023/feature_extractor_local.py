import torch
import os, sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump
from utils.models import load_ground_truth
import argparse
import random
from sklearn.metrics import roc_auc_score

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)


def get_features_and_labels_by_model_class(args, models_dirpath: str):
    ### A: FCModel, B: CNNModel

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

        if model_class == 'FCModel':
            model_A_features.append(model_features)
            model_A_labels.append(model_ground_truth)
        elif model_class == 'CNNModel':
            model_B_features.append(model_features)
            model_B_labels.append(model_ground_truth)
        else:
            print(model_class, " is an unknown model!!!!!!")

    return {'model_A_features': np.asarray(model_A_features), 'model_B_features': np.asarray(model_B_features),
            'model_A_labels': np.asarray(model_A_labels), 'model_B_labels': np.asarray(model_B_labels)}


def get_features_and_labels(args, models_dirpath: str):
    ### A: FCModel, B: CNNModel

    model_A_features, model_B_features, model_C_features, model_A_labels, model_B_labels, model_C_labels = [], [], [], [], [], []

    # model_names = []
    # for model_path in models_dirpath:
    #     if not os.path.isdir(model_path):
    #         continue
    #     model = torch.load(os.path.join(model_path, "model.pt"))
    #     model_name = model._get_name()
    #     model_names.append(model_name)

    for model_path in models_dirpath:
        if not os.path.isdir(model_path):
            continue
        class_and_features_by_model = get_general_model_features(args, os.path.join(model_path, "model.pt"))
        model_class, model_features = class_and_features_by_model['model_class'], class_and_features_by_model['features']
        model_ground_truth = load_ground_truth(model_path)

        # if model_class == 'FCModel':
        #     model_A_features.append(model_features)
        #     model_A_labels.append(model_ground_truth)
        # elif model_class == 'CNNModel':
        #     model_B_features.append(model_features)
        #     model_B_labels.append(model_ground_truth)
        # else:
        #     print(model_class, " is an unknown model!!!!!!")
        model_A_features.append(model_features)
        model_A_labels.append(model_ground_truth)

    return {'model_A_features': np.asarray(model_A_features), 'model_A_labels': np.asarray(model_A_labels)}


def get_general_model_features(args, model_filepath):
    if 'cyber-pdf-dec2022-train' in model_filepath:
        model = torch.load(model_filepath)
        model_class = model._get_name()
        model_backbone = model
    else:
        model, model_repr, model_class = load_model(model_filepath)
        model_backbone = model.model
        # num_of_params = sum(p.numel() for p in model_backbone.parameters()) / 1000.0
        # print(model_class, num_of_params)

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    features = []
    features = _get_general_eigen_vals(all_backbone_params, args['low_layer'], args['high_layer'], args['num_eigen_values'])

    return {'model_class': model_class, 'features': features}


def _get_general_eigen_vals(all_backbone_params, idx_low=0, idx_high=3, num_eigen_values=5):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) >= 2:
            if (num_layers >= idx_low and num_layers <= idx_high) or (backbone_params.shape[0] == 2):
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values**2
                if squared_singular_values.shape[0] >= num_eigen_values:
                    top_five_sq_sv = squared_singular_values[:num_eigen_values]
                    #features += squared_singular_values.tolist()
                    features += top_five_sq_sv.tolist()
                    num_layers += 1
                elif squared_singular_values.shape[0] == 2:
                    top_five_sq_sv = squared_singular_values[:num_eigen_values]
                    # features += squared_singular_values.tolist()
                    features += top_five_sq_sv.tolist()
            #num_layers += 1
    #print(num_layers)
    return features


def get_model_features(args, model_filepath):
    model, model_repr, model_class = load_model(model_filepath)
    model_class = model._get_name()
    model_backbone = model

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    features = []

    if model_class == 'FCModel':
        features = _get_eigen_vals_fc(all_backbone_params, args['low_layer'], args['high_layer'])
    elif model_class == 'CNNModel':
        features = _get_eigen_vals(all_backbone_params, args['low_layer'], args['high_layer'])
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
                top_five_sq_sv = squared_singular_values[:5]
                #features += squared_singular_values.tolist()
                features += top_five_sq_sv.tolist()
                # num_layers += 1
            num_layers += 1
        if len(backbone_params.shape) == 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values**2
                top_five_sq_sv = squared_singular_values[:5]
                #features += squared_singular_values.tolist()
                features += top_five_sq_sv.tolist()
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
                top_five_sq_sv = squared_singular_values[:5]
                #features += squared_singular_values.tolist()
                features += top_five_sq_sv.tolist()
                # num_layers += 1
            num_layers += 1
    return features


def get_model_path_list(models_dir_list):
    models_path_list = []
    for models_dir in models_dir_list:
        for model in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, model)):
                models_path_list.append(os.path.join(models_dir, model))

    return models_path_list


def check_model_arch(models_path_list):
    model_classes = []
    models = []
    for model_dir in models_path_list:
        model, model_repr, model_class = load_model(os.path.join(model_dir, "model.pt"))
        model_class = model._get_name()
        # model_backbone = model
        model_classes.append(model_class)
        models.append(model)
    return models, model_classes


if __name__ == "__main__":
    np.random.seed(0)
    # parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    # args = parser.parse_args()
    args = {}
    args['low_layer'] = 0
    args['high_layer'] = 2
    args['num_eigen_values'] = 100

    models_dir_list = ["/scr2/lu/TrojAI23/rl-lavaworld-jul2023/training/models",
                       "/scr2/lu/TrojAI23/rl-lavaworld-jul2023/leftovers/models",
                       "/scr2/lu/TrojAI23/rl-randomized-lavaworld-aug2023-train-dataset/revision1/models",
                       "/scr2/lu/TrojAI23/rl-randomized-lavaworld-aug2023-train-dataset/models"]
    models_dir_list = ["/scr2/lu/TrojAI23/cyber-apk-nov2023-train/models",
                       "/scr2/lu/TrojAI23/cyber-apk-nov2023-train-rev2/models",
                       "/scr2/lu/TrojAI23/cyber-pdf-dec2022-train/models", ]
    models_dir_list = ["/scr2/lu/TrojAI23/cyber-apk-nov2023-train/models",
                       "/scr2/lu/TrojAI23/cyber-apk-nov2023-train-rev2/models",]
    models_path_list = sorted(get_model_path_list(models_dir_list))
    # model_repr_dict, model_ground_truth_dict = load_models_dirpath(models_path_list)
    #
    # models_padding_dict = create_models_padding(model_repr_dict)
    #
    # for model_class, model_repr_list in model_repr_dict.items():
    #     for index, model_repr in enumerate(model_repr_list):
    #         model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)
    #
    # check_models_consistency(model_repr_dict)

    # # Build model layer map to know how to flatten
    # # logging.info("Generating model layer map...")
    # model_layer_map = create_layer_map(model_repr_dict)
    #
    # # Flatten models
    # flat_models = flatten_models(model_repr_dict, model_layer_map)
    # del model_repr_dict
    # # logging.info("Models flattened. Fitting feature reduction...")
    #
    # layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, input_features)


    #all_features_and_labels = get_features_and_labels_by_model_class(args, models_path_list)
    all_features_and_labels = get_features_and_labels(args, models_path_list)
    model_A_features, model_A_labels = all_features_and_labels['model_A_features'], all_features_and_labels['model_A_labels']
    #model_B_features, model_B_labels = all_features_and_labels['model_B_features'], all_features_and_labels['model_B_labels']

    np.save('features_A', model_A_features)
    #np.save('features_B', model_B_features)
    np.save('labels_A', model_A_labels)
    #np.save('labels_B', model_B_labels)

    base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss', bootstrap=True,
                                      random_state=0)
    clf_A, clf_B, clf_C = CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(
        base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5)
    clf_A.fit(model_A_features, model_A_labels)
    #clf_B.fit(model_B_features, model_B_labels)

    dump(clf_A, 'classifier_model_A.joblib')
    #dump(clf_B, 'classifier_model_B.joblib')

    # shuffle_index_A = np.arange(model_A_features.shape[0])
    # np.random.shuffle(shuffle_index_A)
    # np.random.seed(0)
    # shuffle_index_B = np.arange(model_B_features.shape[0])
    # np.random.shuffle(shuffle_index_B)
    #
    # train_index_A = sorted(shuffle_index_A[:int(0.8 * model_A_features.shape[0])])
    # print(train_index_A)
    # val_index_A = sorted(shuffle_index_A[int(0.8 * model_A_features.shape[0]):])
    # print(val_index_A)
    # train_index_B = sorted(shuffle_index_B[:int(0.8 * model_B_features.shape[0])])
    # print(train_index_B)
    # val_index_B = sorted(shuffle_index_B[int(0.8 * model_B_features.shape[0]):])
    # print(val_index_B)
    #
    # train_model_A_features = model_A_features[train_index_A, :]
    # train_model_A_labels = model_A_labels[train_index_A]
    # val_model_A_features = model_A_features[val_index_A, :]
    # val_model_A_labels = model_A_labels[val_index_A]
    # train_model_B_features = model_B_features[train_index_B, :]
    # train_model_B_labels = model_B_labels[train_index_B]
    # val_model_B_features = model_B_features[val_index_B, :]
    # val_model_B_labels = model_B_labels[val_index_B]
    #
    # base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss',  bootstrap=True, random_state=0)
    # clf_A, clf_B, clf_C = CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5)
    # # clf_A.fit(model_A_features, model_A_labels)
    # clf_A.fit(train_model_A_features, train_model_A_labels)
    # clf_B.fit(model_B_features, model_B_labels)
    # # clf_B.fit(train_model_B_features, train_model_B_labels)
    #
    # # dump(clf_A, 'classifier_model_A_'+str(args.low_layer)+str(args.high_layer)+'.joblib')
    # dump(clf_B, 'classifier_model_B_'+str(args.low_layer)+str(args.high_layer)+'.joblib')
    #
    # #### validation
    # probabilities = clf_A.predict_proba(val_model_A_features)
    # auc_score_A = roc_auc_score(val_model_A_labels, probabilities[:, 1])
    # print(auc_score_A)
    # probabilities = clf_B.predict_proba(val_model_B_features)
    # auc_score_B = roc_auc_score(val_model_B_labels, probabilities[:, 1])
    # print(auc_score_B)



