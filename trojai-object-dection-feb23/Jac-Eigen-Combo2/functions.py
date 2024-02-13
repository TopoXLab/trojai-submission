'''
wrap up basic functions in this file.

'''





import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")


def prepare_boxes(anns, image_id):
    '''
    Pre-Inference stage.
    prepare bounding box and labels before feed into models.
    '''
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for answer in anns:
            boxes.append(answer['bbox'])
            class_ids.append(answer['category_id'])

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
        # convert [x,y,w,h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target




def configure(source_dataset_dirpath,
              output_parameters_dirpath,
              configure_models_dirpath,
              parameter3):
    import logging
    import os
    import jsonpickle

    logging.info('Using parameter3 = {}'.format(str(parameter3)))

    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    logging.info('Reading source dataset from ' + source_dataset_dirpath)

    arr = np.random.rand(100,100)
    np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2
    example_dict['keyb'] = 3
    example_dict['keyc'] = 5
    example_dict['keyd'] = 7
    example_dict['keye'] = 11
    example_dict['keyf'] = 13
    example_dict['keyg'] = 17

    with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(example_dict, warn=True, indent=2))





######################################
# Jacobian 
#
#
#######################################

# from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

def iter_gradients(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield x.grad.data
    else:
        for elem in x:
            for result in iter_gradients(elem):
                yield result

def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()


def compute_jacobian(inputs, output):
    """
    original version
    backpropagate function
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs[0].requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()
    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        # https://pytorch.org/docs/1.4.0/autograd.html?highlight=backward#torch.autograd.backward
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data
    # print("jacobian", jacobian.size())

    return torch.transpose(jacobian, dim0=0, dim1=1)



def compute_object_detection_jacobian_ssd(inputs, output):
    """
    backpropagate function
    :param inputs[0]: Size (e.g. Depth X Width X Height) - (3, 478, 640)
    :param output: (1, 8732, 91)
    :return: jacobian: Classes X Size - (91, 3, 478, 640)
    """
    assert inputs[0].requires_grad
    num_classes = output.size()[-1]

    jacobian = torch.zeros(num_classes, *inputs[0].size())
    grad_output = torch.zeros(*output.size())
    if inputs[0].is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()
    for i in range(num_classes):
        grad_output.zero_()
        grad_output[0, :, i] = 1
        # https://pytorch.org/docs/1.4.0/autograd.html?highlight=backward#torch.autograd.backward
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs[0].grad.data
        zero_gradients(inputs[0])

    return jacobian # torch.transpose(jacobian, dim0=0, dim1=1)


def compute_object_detection_jacobian_fasterrcnn(inputs, output):
    """
    backpropagate function
    :param inputs[0]: Size (e.g. Depth X Width X Height) - (3, 478, 640)
    :param output: (372, 91)
    :return: jacobian: Classes X Size - (91, 3, 478, 640)
    """
    assert inputs[0].requires_grad
    num_classes = output.size()[-1]

    jacobian = torch.zeros(num_classes, *inputs[0].size())
    grad_output = torch.zeros(*output.size())
    if inputs[0].is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()
    for i in range(num_classes):
        grad_output.zero_()
        grad_output[:, i] = 1
        # https://pytorch.org/docs/1.4.0/autograd.html?highlight=backward#torch.autograd.backward
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs[0].grad.data
        zero_gradients(inputs[0])

    return jacobian # torch.transpose(jacobian, dim0=0, dim1=1)



def normal_distribution(device):
    '''
    Geneerate normal distribution samples
    '''
    g_cpu = torch.Generator()
    g_cpu.manual_seed(10086)
    # data = torch.normal(mu, sigma, generator = g_cpu, size=(sample_num, embedding_dim)) # get normal distribution with size (sample_num, embedding_dim)
    # data = Variable(data).view(batch_size, 1, -1).to(device)
    # data.requires_grad_(True) # (sample_num, 1, embedding_dim)


    # ((3, 478, 640)
    X=torch.rand((3, 12, 16), generator = g_cpu, requires_grad = True,  dtype = torch.float) # COCO image size
    X.data*=255.
    X = Variable(X).to(device)
    X.requires_grad_(True)

    return X


def compute_metrics(y_test, y_pred):
    '''
    Get metrics. 
    '''
    import sklearn
    import numpy as np
    from sklearn.metrics import accuracy_score

    # y_test, y_pred should be 1D array
    y_test, y_pred = np.array( y_test ), np.array( y_pred )
    acc= accuracy_score(y_test, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    recall=sklearn.metrics.recall_score(y_test, y_pred)
    precision=sklearn.metrics.precision_score(y_test, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)

    print('Label positive {}, Label negative {}, Total {}'.format( np.sum(y_test==1), np.sum(y_test==0), len(y_test)  ) )
    print('acc {:.4f}, auc {:.4f}, recall {:.4f}, precision {:.4f}, F1 {:.4f}, \ncm {}'.format(acc, auc, recall, precision, f1, cm)  )
    
    return acc, auc, recall, precision, f1, cm


def train_rf_randomsearch(_single_fea, labels):
    '''
    Randomized Search CV to train RF classifier. Get a rough range of parameters.
    Input: feas and labels
    Output: auc, acc.
    
    '''
    from sklearn.model_selection import train_test_split 
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    X_train, X_test, y_train, y_test = train_test_split(_single_fea, labels, test_size = 0.2, random_state = 42)
    # norm
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train) # 115, 20
    X_test = scaler.transform(X_test)
    
    ## random search to narrow down the parameter range

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 100)] #+ [5, 10, 100]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(6, 110, num = 11)] #+ [1,2,5]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, n_jobs = 20, random_state=42)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    print('best params for randomizedsearchCV\n', rf_random.best_params_)

    return rf_random.best_estimator_, X_train, X_test, y_train, y_test





def train_rf_randomsearch_final(_single_fea, labels):
    '''
    use all data
    Randomized Search CV to train RF classifier. Get a rough range of parameters.
    Input: feas and labels
    Output: auc, acc.
    
    '''
    from sklearn.model_selection import train_test_split 
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    # norm
    scaler = MinMaxScaler()
    _single_fea = scaler.fit_transform(_single_fea) # 115, 20
    ## random search to narrow down the parameter range

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 100)] #+ [5, 10, 100]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(60, 110, num = 11)] #+ [1,2,5]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, n_jobs = 20, random_state=42)
    # Fit the random search model
    rf_random.fit(_single_fea, labels)

    print('best params for randomizedsearchCV\n', rf_random.best_params_)

    return rf_random.best_estimator_
