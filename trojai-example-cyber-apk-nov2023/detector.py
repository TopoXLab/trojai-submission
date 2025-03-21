import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename
from joblib import dump, load

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import feature_extractor_local as fe

from tqdm import tqdm

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

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        # self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        # self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        # self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        # self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")
        #
        # # TODO: Update skew parameters per round
        # self.model_skew = {
        #     "__all__": metaparameters["infer_cyber_model_skew"],
        # }
        #
        # self.input_features = metaparameters["train_input_features"]
        # self.weight_table_params = {
        #     "random_seed": metaparameters["train_weight_table_random_state"],
        #     "mean": metaparameters["train_weight_table_params_mean"],
        #     "std": metaparameters["train_weight_table_params_std"],
        #     "scaler": metaparameters["train_weight_table_params_scaler"],
        # }
        # self.random_forest_kwargs = {
        #     "n_estimators": metaparameters[
        #         "train_random_forest_regressor_param_n_estimators"
        #     ],
        #     "criterion": metaparameters[
        #         "train_random_forest_regressor_param_criterion"
        #     ],
        #     "max_depth": metaparameters[
        #         "train_random_forest_regressor_param_max_depth"
        #     ],
        #     "min_samples_split": metaparameters[
        #         "train_random_forest_regressor_param_min_samples_split"
        #     ],
        #     "min_samples_leaf": metaparameters[
        #         "train_random_forest_regressor_param_min_samples_leaf"
        #     ],
        #     "min_weight_fraction_leaf": metaparameters[
        #         "train_random_forest_regressor_param_min_weight_fraction_leaf"
        #     ],
        #     "max_features": metaparameters[
        #         "train_random_forest_regressor_param_max_features"
        #     ],
        #     "min_impurity_decrease": metaparameters[
        #         "train_random_forest_regressor_param_min_impurity_decrease"
        #     ],
        # }

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # for random_seed in np.random.randint(1000, 9999, 10):
        #     self.weight_table_params["random_seed"] = random_seed
        #     self.manual_configure(models_dirpath)
        return self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # get hyperparameters
        args = {}
        args['low_layer'] = 0
        args['high_layer'] = 0
        args['num_eigen_values'] = 100

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        # model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)
        #
        # models_padding_dict = create_models_padding(model_repr_dict)
        # with open(self.models_padding_dict_filepath, "wb") as fp:
        #     pickle.dump(models_padding_dict, fp)
        #
        # for model_class, model_repr_list in model_repr_dict.items():
        #     for index, model_repr in enumerate(model_repr_list):
        #         model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)
        #
        # check_models_consistency(model_repr_dict)
        #
        # # Build model layer map to know how to flatten
        # logging.info("Generating model layer map...")
        # model_layer_map = create_layer_map(model_repr_dict)
        # with open(self.model_layer_map_filepath, "wb") as fp:
        #     pickle.dump(model_layer_map, fp)
        # logging.info("Generated model layer map. Flattenning models...")
        #
        # # Flatten models
        # flat_models = flatten_models(model_repr_dict, model_layer_map)
        # del model_repr_dict
        # logging.info("Models flattened. Fitting feature reduction...")
        #
        # layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)

        all_features_and_labels = fe.get_features_and_labels(args, model_path_list)
        model_A_features, model_A_labels = all_features_and_labels['model_A_features'], all_features_and_labels[
            'model_A_labels']
        # model_B_features, model_B_labels = all_features_and_labels['model_B_features'], all_features_and_labels[
        #     'model_B_labels']

        np.save('features_A', model_A_features)
        # np.save('features_B', model_B_features)
        np.save('labels_A', model_A_labels)
        # np.save('labels_B', model_B_labels)

        base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss', bootstrap=True,
                                          random_state=0)
        clf_A, clf_B, clf_C = CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(
            base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5)
        clf_A.fit(model_A_features, model_A_labels)
        # clf_B.fit(model_B_features, model_B_labels)

        dump(clf_A, os.path.join(self.learned_parameters_dirpath, 'classifier_model_A.joblib'))
        # dump(clf_B, os.path.join(self.learned_parameters_dirpath, 'classifier_model_B.joblib'))

        # logging.info("Feature reduction applied. Creating feature file...")
        # X = None
        # y = []
        #
        # for _ in range(len(flat_models)):
        #     (model_arch, models) = flat_models.popitem()
        #     model_index = 0
        #
        #     logging.info("Parsing %s models...", model_arch)
        #     for _ in tqdm(range(len(models))):
        #         model = models.pop(0)
        #         y.append(model_ground_truth_dict[model_arch][model_index])
        #         model_index += 1
        #
        #         model_feats = use_feature_reduction_algorithm(
        #             layer_transform[model_arch], model
        #         )
        #         if X is None:
        #             X = model_feats
        #             continue
        #
        #         X = np.vstack((X, model_feats * self.model_skew["__all__"]))
        #
        # logging.info("Training RandomForestRegressor model...")
        # model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        # model.fit(X, y)
        #
        # logging.info("Saving RandomForestRegressor model...")
        # with open(self.model_filepath, "wb") as fp:
        #     pickle.dump(model, fp)
        #
        # self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        inputs_np = None
        g_truths = []

        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                if not os.path.exists(ground_truth_filename):
                    logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                    continue

                new_input = np.load(examples_dir_entry.path)

                if inputs_np is None:
                    inputs_np = new_input
                else:
                    inputs_np = np.concatenate([inputs_np, new_input])

                with open(ground_truth_filename) as f:
                    data = int(json.load(f))

                g_truths.append(data)

        g_truths_np = np.asarray(g_truths)

        p = model.predict(inputs_np)

        orig_test_acc = accuracy_score(g_truths_np, np.argmax(p.detach().numpy(), axis=1))
        print("Model accuracy on example data {}: {}".format(examples_dirpath, orig_test_acc))


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # with open(self.model_layer_map_filepath, "rb") as fp:
        #     model_layer_map = pickle.load(fp)
        #
        # # List all available model and limit to the number provided
        # model_path_list = sorted(
        #     [
        #         join(round_training_dataset_dirpath, 'models', model)
        #         for model in listdir(join(round_training_dataset_dirpath, 'models'))
        #     ]
        # )
        # logging.info(f"Loading %d models...", len(model_path_list))
        #
        # model_repr_dict, _ = load_models_dirpath(model_path_list)
        # logging.info("Loaded models. Flattenning...")
        #
        # with open(self.models_padding_dict_filepath, "rb") as fp:
        #     models_padding_dict = pickle.load(fp)
        #
        # for model_class, model_repr_list in model_repr_dict.items():
        #     for index, model_repr in enumerate(model_repr_list):
        #         model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)
        #
        # # Flatten model
        # flat_models = flatten_models(model_repr_dict, model_layer_map)
        # del model_repr_dict
        # logging.info("Models flattened. Fitting feature reduction...")
        #
        # layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)
        #
        # model, model_repr, model_class = load_model(model_filepath)
        # model_repr = pad_model(model_repr, model_class, models_padding_dict)
        # flat_model = flatten_model(model_repr, model_layer_map[model_class])

        # get hyperparameters
        args = {}
        args['low_layer'] = 0
        args['high_layer'] = 0
        args['num_eigen_values'] = 100

        # predict_model_class_and_features = fe.get_model_features(args, model_filepath)
        predict_model_class_and_features = fe.get_general_model_features(args, model_filepath)
        predict_model_class = predict_model_class_and_features['model_class']
        predict_model_features = np.asarray([predict_model_class_and_features['features']])
        # if predict_model_class == 'FCModel':
        #     clf = load('/learned_parameters/classifier_model_A.joblib')
        #     probability = clf.predict_proba(predict_model_features)
        # elif predict_model_class == 'CNNModel':
        #     clf = load('/learned_parameters/classifier_model_B.joblib')
        #     probability = clf.predict_proba(predict_model_features)
        # else:
        #     logging.warning('No able to detect such model class')
        #     probability = [0.5, 0.5]
        clf = load('/learned_parameters/classifier_model_A.joblib')
        probability = clf.predict_proba(predict_model_features)
        logging.info('Trojan Probability of this class {} model is: {}'.format(predict_model_class, probability))

        # # Inferences on examples to demonstrate how it is done for a round
        # # This is not needed for the random forest classifier
        # self.inference_on_example_data(model, examples_dirpath)
        #
        # X = (
        #     use_feature_reduction_algorithm(layer_transform[model_class], flat_model)
        #     * self.model_skew["__all__"]
        # )
        #
        # try:
        #     with open(self.model_filepath, "rb") as fp:
        #         regressor: RandomForestRegressor = pickle.load(fp)
        #
        #     probability = str(regressor.predict(X)[0])
        # except Exception as e:
        #     logging.info('Failed to run regressor, there may have an issue during fitting, using random for trojan probability: {}'.format(e))
        #     probability = str(np.random.rand())
        # with open(result_filepath, "w") as fp:
        #     fp.write(probability)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability[0, -1]))

        logging.info("Trojan probability: {}".format(probability[0, -1]))
        return probability[0, -1]
