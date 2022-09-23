from inference import inference
from evaluation.evaluate import evaluate


def valid(model, data_loader_test, pred_dir, method='tmp_val', testset='DIS-VD', only_S_MAE=True):
    model_training = model.training
    if model_training:
        model.eval()
    inference(model, data_loader_test, pred_dir, method, testset)
    performance_dict = evaluate(pred_dir, method, testset, only_S_MAE=only_S_MAE, epoch=model.epoch)
    if model_training:
        model.train()
    return performance_dict
