# todo: This file contains all the xai methods in functios

from openxai.experiment_utils import fill_param_dict
from openxai.explainer import Explainer
import warnings; warnings.filterwarnings("ignore")

def get_lime_feature_importances(model, X_train, inputs, preds):
    # LIME gradient

    method = 'lime'

    # Pass empty dict to use default parameters
    param_dict = {}

    # If LIME/IG, then provide X_train
    param_dict = fill_param_dict(method, {}, X_train)
    params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in
                      param_dict.items()]
    print(f'{method.upper()} Parameters\n\n' + '\n'.join(params_preview))
    print('Remaining parameters are set to their default values')

    # Compute explanations
    lime = Explainer(method, model, param_dict)
    lime_exps = lime.get_explanations(inputs.float(), preds).detach().numpy()
    # print(lime_exps[0])

    return lime_exps

def get_ig_feature_importances(model, X_train, inputs, preds):
    # INTEGRATED GRADIENT method
    method = 'ig'

    # If LIME/IG, then provide X_train
    param_dict = fill_param_dict('ig', {}, X_train)

    # Compute explanations
    ig = Explainer(method, model, param_dict)
    ig_exps = ig.get_explanations(inputs.float(), preds).detach().numpy()
    # print(ig_exps[0])

    return ig_exps

def get_shap_feature_importances(model, inputs, preds):
    # SHAP method

    method = 'shap'
    param_dict = {'n_samples': 1000, 'seed': 0}

    # Compute explanations
    shap = Explainer(method, model, param_dict)
    shap_exps = shap.get_explanations(inputs.float(), preds).detach().numpy()
    # print(shap_exps[0])

    return shap_exps

def get_sg_feature_importances(model, inputs, preds):
    # SMOOTH GRADIENT method

    method = 'sg'

    # Override default parameters for certain hyperparameters
    param_dict = {'n_samples': 1000, 'seed': 0}

    # Compute explanations
    sg = Explainer(method, model, param_dict)
    sg_exps = sg.get_explanations(inputs.float(), preds).detach().numpy()
    # print(sg_exps[0])

    return sg_exps

def get_itg_feature_importances(model, inputs, preds):
    # INPUT TIMES GRADIENT method

    method = 'itg'

    # Override default parameters for certain hyperparameters
    param_dict = {'n_samples': 1000, 'seed': 0}

    # Compute explanations
    itg = Explainer(method, model, {})
    itg_exps = itg.get_explanations(inputs.float(), preds).detach().numpy()
    # print(itg_exps[0])

    return itg_exps

def get_grad_feature_importances(model, inputs, preds):
    # GRADIENT method

    method = 'grad'

    # Override default parameters for certain hyperparameters
    param_dict = {'n_samples': 1000, 'seed': 0}

    # Compute explanations
    grad = Explainer(method, model, {})
    grad_exps = grad.get_explanations(inputs.float(), preds).detach().numpy()
    # print(grad_exps[0])

    return grad_exps

