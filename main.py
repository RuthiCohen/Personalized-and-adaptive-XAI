# todo: wrap all xai methods in functions
# todo: write a config file with all data of datasets(also in dataloader.py)
# todo: upload files to github :)

import pandas as pd
from openxai.model import train_model

from openxai.experiment_utils import plot_feature_importance, \
get_feature_names, mapping_result, fill_param_dict, split_to_train_test_files

from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
from openxai.explainer import Explainer
import warnings; warnings.filterwarnings("ignore")


file_name = 'heart_failure_clinical_records_dataset'

#if data is not splitted to train&test files - split them
split_to_train_test_files(file_name)

path = f"data/{file_name}/{file_name}.csv"
data = pd.read_csv(path)
model_kind = "ann"

# print(data.head(10))

trainloader, testloader = ReturnLoaders(file_name)
inputs, labels = next(iter(testloader))

X_train, X_test = ReturnTrainTestX(file_name, float_tensor=True)

learning_rate, epochs, batch_size = 0.001, 100, 32
# Load pretrained ml model
model, best_acc, best_epoch = train_model(model_kind, file_name, learning_rate, epochs, batch_size)

preds = model(inputs.float()).argmax(1)
print(f'First 10 predictions: {preds[:10]}')

features = get_feature_names(file_name)
# print(features)

### Explainers ####
#### --- lime --- ###

# Choose explainer
method = 'lime'

# Pass empty dict to use default parameters
param_dict = {}

# # If LIME/IG, then provide X_train
param_dict = fill_param_dict(method, {}, X_train)
params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
print(f'{method.upper()} Parameters\n\n' +'\n'.join(params_preview))
print('Remaining parameters are set to their default values')

# Compute explanations
lime = Explainer(method, model, param_dict)
lime_exps = lime.get_explanations(inputs.float(), preds).detach().numpy()
# print(lime_exps[0])

plot_feature_importance(lime_exps[0], features, method, file_name)
#### ---  -------------- ###
# Choose explainer
method = 'ig'

# If LIME/IG, then provide X_train
param_dict = fill_param_dict('ig', {}, X_train)

# Compute explanations
ig = Explainer(method, model, param_dict)
ig_exps = ig.get_explanations(inputs.float(), preds).detach().numpy()
# print(ig_exps[0])

plot_feature_importance(ig_exps[0], features, method, file_name)

#--------------
# Choose explainer
method = 'shap'

# Override default parameters for certain hyperparameters
param_dict = {'n_samples': 1000, 'seed': 0}

# Compute explanations
shap = Explainer(method, model, param_dict)
shap_exps = shap.get_explanations(inputs.float(), preds).detach().numpy()
# print(shap_exps[0])

plot_feature_importance(shap_exps[0], features, method, file_name)

#-------------------
# Choose explainer
method = 'sg'

# Override default parameters for certain hyperparameters
param_dict = {'n_samples': 1000, 'seed': 0}

# Compute explanations
sg = Explainer(method, model, param_dict)
sg_exps = sg.get_explanations(inputs.float(), preds).detach().numpy()
# print(sg_exps[0])

plot_feature_importance(sg_exps[0], features, method, file_name)

# #--------------------------
# Choose explainer
method = 'itg'

# Override default parameters for certain hyperparameters
param_dict = {'n_samples': 1000, 'seed': 0}

# Compute explanations
itg = Explainer(method, model, {})
itg_exps = itg.get_explanations(inputs.float(), preds).detach().numpy()
# print(itg_exps[0])

plot_feature_importance(itg_exps[0], features, method, file_name)
# # ---------------------------------
# Choose explainer
method = 'grad'

# Override default parameters for certain hyperparameters
param_dict = {'n_samples': 1000, 'seed': 0}

# Compute explanations
grad = Explainer(method, model, {})
grad_exps = grad.get_explanations(inputs.float(), preds).detach().numpy()
# print(grad_exps[0])

plot_feature_importance(grad_exps[0], features, method, file_name)

mapping_result(lime_exps[0],
               ig_exps[0],
               shap_exps[0],
               sg_exps[0],
               itg_exps[0],
               grad_exps[0],
               features)
