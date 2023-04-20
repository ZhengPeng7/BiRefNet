from inference import inference
from evaluation.evaluate import evaluate


def valid(model, data_loader_test, pred_dir, method='tmp_val', testset='DIS-VD', only_S_MAE=True, device=0):
    model.eval()
    inference(model, data_loader_test, pred_dir, method, testset, device=device)
    performance_dict = evaluate(pred_dir, method, testset, only_S_MAE=only_S_MAE, epoch=model.epoch)
    return performance_dict
