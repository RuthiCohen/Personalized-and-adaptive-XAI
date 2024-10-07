from openxai.model import train_model

from openxai.experiment_utils import plot_feature_importance, \
get_feature_names, mapping_result, split_to_train_test_files

from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
from xai_utils import get_lime_feature_importances, get_ig_feature_importances, \
get_shap_feature_importances, get_sg_feature_importances, \
get_grad_feature_importances, get_itg_feature_importances

from preprocess_data import get_preprocessed_data

import warnings; warnings.filterwarnings("ignore")

if __name__ == "__main__":
    file_name = 'heart_failure_clinical_records_dataset'
    # file_name = 'MBA'
    # file_name = "student_performance_factors" #todo: support this data..
    # file_name = "undergraduate_admission_test_survey_in_bangladesh"

    data, path = get_preprocessed_data(file_name)
    model_kind = "ann"

    trainloader, testloader = ReturnLoaders(file_name)
    inputs, labels = next(iter(testloader))

    X_train, X_test = ReturnTrainTestX(file_name, float_tensor=True)

    learning_rate, epochs, batch_size = 0.001, 100, 32
    model, best_acc, best_epoch = train_model(model_kind, file_name, learning_rate, epochs, batch_size)

    preds = model(inputs.float()).argmax(1)
    print(f'First 10 predictions: {preds[:10]}')

    features = get_feature_names(file_name)

    # XAI methods results
    lime_exps = get_lime_feature_importances(model, X_train, inputs, preds)
    plot_feature_importance(lime_exps[0], features, "lime", file_name)

    ig_exps = get_ig_feature_importances(model, X_train, inputs, preds)
    plot_feature_importance(ig_exps[0], features, "ig", file_name)

    shap_exps = get_shap_feature_importances(model, inputs, preds)
    plot_feature_importance(shap_exps[0], features, "shap", file_name)

    sg_exps = get_sg_feature_importances(model, inputs, preds)
    plot_feature_importance(sg_exps[0], features, "sg", file_name)

    itg_exps = get_itg_feature_importances(model, inputs, preds)
    plot_feature_importance(itg_exps[0], features, "itg", file_name)

    grad_exps = get_grad_feature_importances(model, inputs, preds)
    plot_feature_importance(grad_exps[0], features, "grad", file_name)

    # show XAI full table results
    mapping_result(lime_exps[0],
                   ig_exps[0],
                   shap_exps[0],
                   sg_exps[0],
                   itg_exps[0],
                   grad_exps[0],
                   features)






