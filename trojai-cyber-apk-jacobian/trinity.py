import os
import sys
import time
from pathlib import Path
import importlib
import math
import copy
import numpy as np

import torch
import util.db as db
import util.smartparse as smartparse
import helper_cyber_pdf as helper
from util.fix_random import fix_random

#Interface spec
# Interface.model: nn.Module
#   support zero_grad()
#   support named_parameters()

# load_examples: returns a table of examples
# eval: return correctness: loss value of one or multiple examples
BINS = 100

def get_model_features(interface, model_class=None):
    model = interface.model
    all_backbone_params = []
    for param in model.parameters():
        all_backbone_params.append(param.data.cpu().numpy())
    # print("print size of parameters:")
    # for param in model.parameters():
    #     print(param.shape)
    features = _get_eigen_vals(all_backbone_params, 0, 1000, model_class)

    return features


def _get_eigen_vals(all_backbone_params, idx_low=0, idx_high=0, model_class=None):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:

        if len(backbone_params.shape) == 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                # reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(backbone_params.transpose(), False)
                squared_singular_values = singular_values ** 2
                top_five_sq_sv = squared_singular_values[:BINS]
                # param_features = np.hstack((top_five_sq_sv, top_five_sq_sv))
                features.append(top_five_sq_sv)
                #num_layers += 1
            #num_layers += 1
        elif len(backbone_params.shape) > 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values ** 2
                # top_five_sq_sv = squared_singular_values[:5]
                features += squared_singular_values.tolist()
                #num_layers += 1
        elif len(backbone_params.shape) == 1:
            continue
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = np.tile(backbone_params, (int(BINS), int(BINS/backbone_params.shape[0])))
                #reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params, False)
                squared_singular_values = singular_values ** 2
                top_five_sq_sv = squared_singular_values[:BINS]
                param_features = np.hstack((top_five_sq_sv, top_five_sq_sv))
                features.append(param_features)
                # num_layers += 1
        num_layers += 1
        if features[-1].shape[0] != BINS:
            features[-1] = np.tile(features[-1], (int(BINS/features[-1].shape[0])))




    return features


def compute_grad(interface,example):
    interface.model.zero_grad()
    s=interface.eval(example)
    loss=s
    loss.backward()
    params=interface.model.named_parameters();
    g=[];
    for k,p in params:
        #if not p.grad is None:
        g.append(p.grad.data.clone().cpu())
        #print(p.grad.max(),p.grad.min())
        #else:
        #    g.append(None)
    
    return g


def analyze_tensor(w,params=None):
    default_params=smartparse.obj();
    default_params.bins=BINS;
    params=smartparse.merge(params,default_params)
    
    if w is None:
        return torch.Tensor(params.bins*2).fill_(0)
    else:
        q=torch.arange(params.bins).float().cuda()/(params.bins-1)
        hw=torch.quantile(w.view(-1).float(),q.to(w.device)).contiguous().cpu();
        hw_abs=torch.quantile(w.abs().view(-1).float(),q.to(w.device)).contiguous().cpu();
        fv=torch.cat((hw,hw_abs),dim=0);
        return fv;


def grad2fv(g,params=None):
    fvs=[analyze_tensor(w,params) for w in g]
    fvs=torch.stack(fvs,dim=0);
    return fvs;


def characterize(interface,data=None,params=None):
    if data is None:
        data=interface.load_examples()
    
    fvs=[grad2fv(compute_grad(interface,data[i]),params) for i in range(len(data))]
    # fvs = []
    # weight_features = get_model_features(interface)
    # weight_feat = torch.from_numpy(np.vstack(weight_features))
    # fvs.append(weight_feat)
    fvs=torch.stack(fvs, dim=0);
    #fvs = torch.stack(fvs, weight_feat)
    print(fvs.shape)
    return {'fvs':fvs}


def extract_fv(interface,ts_engine,params=None):
    data=ts_engine(interface,params=params);
    fvs=characterize(interface,data,params);
    return fvs


#Extract dataset from a folder of training models
def extract_dataset(models_dirpath,ts_engine,params=None):
    default_params=smartparse.obj();
    default_params.rank=0
    default_params.world_size=1
    default_params.out=''
    params=smartparse.merge(params,default_params)
    
    t0=time.time()
    models=os.listdir(models_dirpath);
    models=sorted(models)
    models=[(i,x) for i,x in enumerate(models)]
    
    
    dataset=[];
    for i,fname in models[params.rank::params.world_size]:
        folder=os.path.join(models_dirpath,fname);
        interface=helper.engine(folder=folder,params=params)
        fvs=extract_fv(interface,ts_engine,params=params);
        
        #Load GT
        fname_gt=os.path.join(folder,'ground_truth.csv');
        f=open(fname_gt,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data={'params':vars(params),'model_name':fname,'model_id':i,'label':label};
        fvs.update(data);
        
        if not params.out=='':
            Path(params.out).mkdir(parents=True, exist_ok=True);
            torch.save(fvs,'%s/%d.pt'%(params.out,i));
        
        dataset.append(fvs);
        print('Model %d(%s), time %f'%(i,fname,time.time()-t0));
    
    dataset=db.Table.from_rows(dataset);
    return dataset;


def generate_probe_set(models_dirpath,params=None):
    models=os.listdir(models_dirpath)
    models=sorted(models)
    
    all_data=[];
    for m in models[:len(models)//2]:
        interface=helper.engine(os.path.join(models_dirpath,m))
        data=interface.load_examples()
        data=list(data.rows())
        
        try:
            data2=interface.load_examples(os.path.join(models_dirpath,m,'poisoned-example-data'))
            data2=list(data2.rows())
            data+=data2
        except:
            pass;
        
        all_data+=data;
    
    all_data=db.Table.from_rows(all_data)
    return all_data


def ts_engine(interface, additional_data='enum.pt', params=None):
    data = interface.load_examples()
    labels = set(data['label'])
    print(labels)
    new_data=[];
    for i in range(len(data)):
        for j in labels:
            d=copy.deepcopy(data[i]);
            d['label']=j;
            new_data.append(d);
    
    data=db.Table.from_rows(new_data)
    
    return data


def predict(ensemble,fvs):
    scores=[];
    fvs = fvs.cuda()
    with torch.no_grad():
        for i in range(len(ensemble)):
            params=ensemble[i]['params'];
            arch=importlib.import_module(params.arch);
            net=arch.new(params);
            net.load_state_dict(ensemble[i]['net'], strict=True);
            net=net.cuda();
            net.eval();

            #print(fvs.get_device())
            s=net.logp(fvs);
            s=s*math.exp(-ensemble[i]['T']);
            scores.append(s)
    
    s=sum(scores)/len(scores);
    s=torch.sigmoid(s); #score -> probability
    trojan_probability=float(s);
    return trojan_probability


if __name__ == "__main__":
    fix_random(1003)
    import os
    default_params = smartparse.obj();
    default_params.out = 'data_r12_trinity_v0'
    params = smartparse.parse(default_params);
    params.argv = sys.argv;
    
    extract_dataset(os.path.join(helper.root(), 'models'), ts_engine, params);
    
