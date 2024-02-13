# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
'''
Extract the jacobian features


'''
import numpy as np
import cv2
import torch
import torchvision
import json
import pickle 
import logging
import warnings

warnings.filterwarnings("ignore")

import glob
import sys, os
sys.path.insert(1, os.path.abspath("../")) # load models.py
sys.path.insert(1, os.path.abspath("../utils/"))
from functions import normal_distribution, compute_object_detection_jacobian_ssd, compute_object_detection_jacobian_fasterrcnn



def jacobian_object_detection(model_filepath, model_architecture, device):
    # load or generate normal distribution samples
    distribution_fp = './learned_parameters/normal_distribution.v2.resize.pkl'
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

    samples_jacobian = None

    # gaussian_sample.retains_grad = True
    # iterate all rand samples
    images = [gaussian_sample]

    # a = torch.autograd.functional.jacobian(pytorch_model, images, create_graph=False)
    output = pytorch_model(images)  #head_outputs_cls_logits - torch.Size([1, 8732, 91]) or [372, 91]
    # output = outputs[1] #head_outputs_cls_logits - torch.Size([1, 8732, 91])
    if model_architecture == 'ssd':
        jacobian_mat = compute_object_detection_jacobian_ssd(images, output) #  torch.Size([91, 3, 478, 640])
    else:
        jacobian_mat = compute_object_detection_jacobian_fasterrcnn(images, output) #  torch.Size([91, 3, 478, 640])

    jacobian_mean = jacobian_mat.view(1, -1) # (1, ..) # 83516160
    jacobian_mean = jacobian_mean.cpu().detach().numpy()[0]
    # print('samples_jacobian ([83516160]])', jacobian_mean.shape)


    # samples_jacobian = jacobian_mat.unsqueeze(0) if samples_jacobian is None else torch.vstack((samples_jacobian, jacobian_mat.unsqueeze(0)))
    # print('samples_jacobian ([200, 91, 3, 478, 640]])', samples_jacobian.shape)
    # jacobian_mean = torch.mean(samples_jacobian, 0) # jacobian_mean torch.Size([NO_CLASS, 3, 478, 640])
    # # jacobian_mean = torch.reshape( jacobian_mean, (1, NO_CLASS*embedding_dim)) # ()
    # jacobian_mean = jacobian_mean.view(1, -1) # (1, ..) # 83516160
    # jacobian_mean = jacobian_mean.cpu().detach().numpy()[0]
    return jacobian_mean



if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_root', type=str, 
        default = '/scr/weimin/RoundData/round10/round10-train-dataset', help='Root folder to save all training models.')
    parser.add_argument('--gpus', type=str, help='Specify GPU usage', default='2')
    parser.add_argument('--model_type', type=str, help='ssd or fasterrcnn', default='ssd')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    model_type = args.model_type # 'fasterrcnn'
    logging.basicConfig(filename='train_jacobian_log_{}.v2.txt'.format(model_type),
                        filemode='w',
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("train_jacobian.py launched")
    logging.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    jacobian_feas, label_list = list(), list()

    model_list = sorted(  [ model for model in os.listdir(args.model_root) ] )
    for model_folder in model_list:
        # general config
        model_filepath = os.path.join(args.model_root, model_folder) + '/model.pt'
        config_path = os.path.join(args.model_root, model_folder) + '/config.json'
        with open(config_path) as json_file:
            config = json.load(json_file)
        label = 1 if config['py/state']['poisoned'] else 0 # true, poisoned, 1; false, clean, 0
        model_architecture = config['py/state']['model_architecture'] # ssd / fasterrcnn
        source_dataset = config['py/state']['source_dataset'] # COCO

        if model_architecture != model_type:
            continue

        # trojan related config
        if label == 1: # trojan
            source_class = config['py/state']['trigger']['py/state']['source_class']
            target_class = config['py/state']['trigger']['py/state']['target_class']
            source_class_label = config['py/state']['trigger']['py/state']['source_class_label']['name']
            target_class_label = config['py/state']['trigger']['py/state']['target_class_label']['name']

            trigger_size = config['py/state']['trigger']['py/state']['trigger_executor']['py/state']['trigger_size']
            trigger_location = config['py/state']['trigger']['py/state']['trigger_executor']['py/state']['location']
            trigger_type = config['py/state']['trigger']['py/state']['trigger_executor']['py/state']['type'] # misclassification / evasion
            trigger_options = config['py/state']['trigger']['py/state']['trigger_executor']['py/state']['options'] # local / global

            # info 
            logging.info('{}, Trojan {}, model_architecture {}, source_dataset {}, source_class {} - {}, target_class {} - {} \
                \n  trigger size {}, location {}, type {}, option {}'\
                .format(model_folder, label, model_architecture, source_dataset, source_class, source_class_label, target_class, target_class_label, \
                    trigger_size, trigger_location, trigger_type, trigger_options))

        else: # clean
            # info 
            logging.info('{}, Trojan {}, model_architecture {}, source_dataset {}, '.format(model_folder, label, model_architecture, source_dataset))

        examples_dirpath = os.path.join( args.model_root, model_folder,  'clean-example-data')
        # print(examples_dirpath)
        # print(os.path.exists(examples_dirpath))


        # # example_trojan_detector(model_filepath)
        jacob_fea = jacobian_object_detection(model_filepath, model_architecture, device)
        jacobian_feas.append(jacob_fea)
        label_list.append(label)



    jacob_feas_file = '../learned_parameters/jacob_feas_{}.v2.pkl'.format(model_type)

    with open(jacob_feas_file, 'wb') as f:
        pickle.dump([jacobian_feas, label_list], f)
        f.close()

    # with open(jacob_feas_file, 'rb') as f:
    #     [jacobian_feas, label_list] = pickle.load(f)
    #     f.close()


# python example_trojan_detector.py --model_filepath=/scr/weimin/RoundData/round10/round10-train-dataset/id-00000000/model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=/scr/weimin/RoundData/round10/round10-train-dataset/id-00000000/clean-example-data/ --source_dataset_dirpath=/path/to/source/dataset/ --round_training_dataset_dirpath=/path/to/training/dataset/ --metaparameters_filepath=./metaparameters.json --schema_filepath=./metaparameters_schema.json --learned_parameters_dirpath=./learned_parameters