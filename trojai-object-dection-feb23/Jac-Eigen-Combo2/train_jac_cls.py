'''
Sentiment Analysis Task
Baseline model_factories. 
Compute Jacobian matrix from cross entropy loss function. Manually compute jacobian
Search for best combination of mean and var of gaussian distribution.


'''

import numpy as np
from scipy.fftpack import ss_diff
import sklearn
import torch
import pickle
import sys, os
sys.path.insert(1, os.path.abspath("../")) # load models.py
sys.path.insert(1, os.path.abspath("../utils/"))
from functions import prepare_boxes, normal_distribution, compute_object_detection_jacobian_ssd, compute_object_detection_jacobian_fasterrcnn
from functions import train_rf_randomsearch, compute_metrics, train_rf_randomsearch_final


import logging
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_type', type=str, 
        default = 'fasterrcnn')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type #'ssd' # 'fasterrcnn'
    jacob_feas_file = '../learned_parameters/jacob_feas_{}.v2.pkl'.format(model_type)
    with open(jacob_feas_file, 'rb') as f:
        [jacobian_feas, label_list] = pickle.load(f)
        f.close()


    acc_list, auc_list = [], []


    jacobian_feas = np.asarray(jacobian_feas) # (model_num, 83516160)
    label_list = np.array(label_list)
    print('jacobian_feas', jacobian_feas.shape)
    print('label_list', label_list.shape)


    # # # # RF, test
    # best_rf, X_train, X_test, y_train, y_test = train_rf_randomsearch(jacobian_feas, label_list)
    # y_pred = best_rf.predict(X_test)
    # train_predict = best_rf.predict(X_train)
    # acc_val, auc_val, recall, precision, f1, cm = compute_metrics(y_test, y_pred)
    # print('TEST PERFORMANCES:')
    # print('   acc {:.4f}, auc {:.4f}, recall {:.4f}, precision {:.4f}, f1 {:.4f}, cm {}'.format( acc_val, auc_val, recall, precision, f1, cm ) )
    # acc, auc, recall, precision, f1, cm = compute_metrics(y_train, train_predict)        
    # print('Train PERFORMANCES:')
    # print('   acc {:.4f}, auc {:.4f}, recall {:.4f}, precision {:.4f}, f1 {:.4f}, cm {}'.format( acc, auc, recall, precision, f1, cm ) )


    # filename = '../learned_parameters/jac_rf_{}.partial.v2.sav'.format(model_type)
    # pickle.dump(best_rf, open(filename, 'wb'))
    

    # # RF, final 
    best_rf = train_rf_randomsearch_final(jacobian_feas, label_list)
    filename = '../learned_parameters/jac_rf_{}_final.v2.sav'.format(model_type)
    pickle.dump(best_rf, open(filename, 'wb'))
    


