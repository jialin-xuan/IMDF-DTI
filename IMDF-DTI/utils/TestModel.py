import torch
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import average_precision_score
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def test_IMDF_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):

    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()

    test_losses = []
    Y, P, S = [], [], []

    with torch.no_grad():
        for i, data in pbar:
            '''数据准备：直接将整个 Batch 移动到设备'''

            data = data.to(DEVICE)


            if isinstance(MODEL, list):


                predicted_scores = torch.zeros(len(data.cls_y), 2).to(DEVICE)
                for m_idx in range(len(MODEL)):

                    predicted_scores = predicted_scores + MODEL[m_idx](data)
                predicted_scores = predicted_scores / FOLD_NUM
            else:

                predicted_scores = MODEL(data)


            labels = data.cls_y


            loss = LOSS(predicted_scores, labels)


            correct_labels = labels.to('cpu').data.numpy()
            predicted_probs = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_probs, axis=1)
            positive_scores = predicted_probs[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(positive_scores)
            test_losses.append(loss.item())


    Precision = precision_score(Y, P, zero_division=0)
    Recall = recall_score(Y, P, zero_division=0)
    AUC = roc_auc_score(Y, S)
    PRC = average_precision_score(Y, S)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)


    return Y, S, np.average(test_losses), Accuracy, Precision, Recall, AUC, PRC


def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):

    return test_IMDF_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM)


def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1, IMDF=False, MIF=None):
    # Accept the old keyword without changing existing callers' behavior.
    if MIF is not None:
        IMDF = MIF
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))


    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_IMDF_precess(MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM) if IMDF else test_precess(MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)


    if save:
        if FOLD_NUM == 1:
            filepath = os.path.join(save_path, "{}_{}_prediction.txt".format(DATASET, dataset_class))
        else:
            filepath = os.path.join(save_path, "{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class))

        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')

    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)


    if IMDF and FOLD_NUM == 1 and dataset_class == "Test":
        try:

            if isinstance(MODEL, list):
                print("Notice: Interpretability maps are generated from the first model in ensemble.")
                target_model = MODEL[0]
            else:
                target_model = MODEL

            print(f"🚀 Generating interpretability reports for {DATASET}...")


            importance_weights = target_model.get_explainability_weights()


            interp_path = os.path.join(save_path, "interpretability_report")


            from utils.Visualization import plot_feature_importance
            plot_feature_importance(importance_weights, save_path=interp_path)

            print(f"✅ Interpretability reports (Heatmaps) saved to: {interp_path}")

        except Exception as e:
            print(f"⚠️ Interpretability output skipped or failed: {e}")

    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


test_MIF_precess = test_IMDF_precess
