"""IMDF-DTI training, retaining the supplied optimization/evaluation logic."""

import os
import random
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import IMDFDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping


from LossFunction import CELoss, PolyLoss, get_mask_l1_loss
from utils.TestModel import test_model
from utils.ShowResult import show_result
from utils import protein_init, ligand_init, ProteinMoleculeDataset
import torch_geometric.loader as pyg_loader

import json
import csv


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_IMDF_model(SEED, DATASET, MODEL, K_Fold, LOSS, device, ABLATION='full', OUTPUT_DIR=None):
    '''设置随机种子'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''初始化超参数'''
    hp = hyperparameter()

    ablation_map = {
        'full': {
            'use_xmask': True,
            'use_dfgu': True,
            'use_rescal_hybrid': True,
            'use_staged_optimization': True,
            'save_root': './results/full'
        },
        'wo_XMask': {
            'use_xmask': False,
            'use_dfgu': True,
            'use_rescal_hybrid': True,
            'use_staged_optimization': True,
            'save_root': './results/wo_XMask'
        },
        'wo_DFGU': {
            'use_xmask': True,
            'use_dfgu': False,
            'use_rescal_hybrid': True,
            'use_staged_optimization': True,
            'save_root': './results/wo_DFGU'
        },
        'wo_RESCAL': {
            'use_xmask': True,
            'use_dfgu': True,
            'use_rescal_hybrid': False,
            'use_staged_optimization': True,
            'save_root': './results/wo_RESCAL'
        },
        'wo_staged_opt': {
            'use_xmask': True,
            'use_dfgu': True,
            'use_rescal_hybrid': True,
            'use_staged_optimization': False,
            'save_root': './results/wo_staged_opt'
        },
    }

    if ABLATION not in ablation_map:
        raise ValueError(f"Unknown ablation setting: {ABLATION}")

    for k, v in ablation_map[ABLATION].items():
        setattr(hp, k, v)

    output_root = OUTPUT_DIR or (DATASET if ABLATION == 'full' else f'{DATASET}_{ABLATION}')
    hp.save_root = str(output_root)

    print(f"[Ablation] {ABLATION}")
    print(f"  use_xmask={hp.use_xmask}")
    print(f"  use_dfgu={hp.use_dfgu}")
    print(f"  use_rescal_hybrid={hp.use_rescal_hybrid}")
    print(f"  use_staged_optimization={hp.use_staged_optimization}")
    print(f"  save_root={hp.save_root}")

    '''从文本文件加载数据集'''
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''设置损失函数权重'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    if DATASET == "BD2D":
        split_pos = 52010
        train_data_list = data_list[:split_pos]
        test_data_list = data_list[split_pos:]
        '''打乱数据'''
        print("data shuffle")
        train_data_list = shuffle_dataset(train_data_list, SEED)
    else:
        '''打乱数据'''
        print("data shuffle")
        data_list = shuffle_dataset(data_list, SEED)

        '''将数据集分割为训练集&验证集和测试集'''
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''数据预处理与加载'''

    os.makedirs('./DataSets/Preprocessed', exist_ok=True)

    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein-new.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi-new.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()

    '''初始化评价指标列表'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []


    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        train_dataset = ProteinMoleculeDataset(train_dataset, ligand_dict, protein_dict, device=device)
        valid_dataset = ProteinMoleculeDataset(valid_dataset, ligand_dict, protein_dict, device=device)
        test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
        train_size = len(train_dataset)

        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

        """ create model"""

        model = MODEL(device=device, hp=hp)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""


        gate_p, base_p = [], []


        if hp.use_staged_optimization and hp.use_dfgu:
            gate_p, base_p = [], []

            for name, p in model.named_parameters():
                if "dfgu" in name.lower() or "gate" in name.lower():
                    gate_p.append(p)
                else:
                    base_p.append(p)

            optimizer = optim.AdamW(
                [
                    {'params': base_p, 'weight_decay': hp.weight_decay, 'lr': hp.Learning_rate},
                    {'params': gate_p, 'weight_decay': hp.weight_decay, 'lr': hp.Learning_rate * hp.gate_lr_ratio},
                ]
            )

            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=[hp.Learning_rate, hp.Learning_rate * hp.gate_lr_ratio],
                max_lr=[hp.Learning_rate * 10, hp.Learning_rate * hp.gate_lr_ratio * 10],
                cycle_momentum=False,
                step_size_up=max(1, train_size // hp.Batch_size)
            )
        else:

            optimizer = optim.AdamW(
                model.parameters(),
                lr=hp.Learning_rate,
                weight_decay=hp.weight_decay
            )

            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=hp.Learning_rate,
                max_lr=hp.Learning_rate * 10,
                cycle_momentum=False,
                step_size_up=max(1, train_size // hp.Batch_size)
            )
        # Kept exactly as supplied. LOSS was not used to select the loss here.
        # Do not switch to PolyLoss without verifying the experiment version.
        Loss = CELoss(weight_CE=weight_loss, DEVICE=device)

        """Output files"""
        save_path = os.path.join(output_root, str(i_fold+1))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'


        hp_dict = {k: v for k, v in hp.__dict__.items() if not k.startswith('__')}
        with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
            json.dump(hp_dict, f, indent=4)


        csv_file = os.path.join(save_path, 'training_results.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['Epoch', 'Train_Loss', 'Valid_Loss', 'Valid_AUC', 'Valid_PRC', 'Accuracy', 'Drug_Weight_2D', 'Prot_Weight_2D'])

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break

            """train"""
            train_losses_in_epoch = []
            epoch_w2d, epoch_w1d = [], []
            model.train()


            if hp.use_staged_optimization and hp.use_dfgu:
                if epoch <= hp.gate_freeze_epochs:
                    train_gate = False
                else:
                    train_gate = ((epoch - hp.gate_freeze_epochs - 1) % hp.gate_update_interval == 0)

                for name, param in model.named_parameters():
                    if "dfgu" in name.lower() or "gate" in name.lower():
                        param.requires_grad = train_gate
                    else:
                        param.requires_grad = True

                if train_gate:
                    print(f"🔓 Epoch {epoch}: update gate/dfgu")
                else:
                    print(f"🔒 Epoch {epoch}: freeze gate/dfgu")
            else:

                train_gate = True
                for param in model.parameters():
                    param.requires_grad = True
                print(f"🔓 Epoch {epoch}: joint optimization")

            for data in train_loader:
                optimizer.zero_grad()

                data = data.to(device)
                predicted_y= model(data)

                with torch.no_grad():


                    w2d = model.dfgu_2d.latest_weights.detach().cpu().numpy()
                    w1d = model.dfgu_1d.latest_weights.detach().cpu().numpy()
                    epoch_w2d.append(w2d)
                    epoch_w1d.append(w1d)


                pred_loss = Loss(predicted_y, data.cls_y)
                train_loss = pred_loss


                if hp.use_dfgu:
                    w2d_tensor = model.dfgu_2d.latest_weights
                    w1d_tensor = model.dfgu_1d.latest_weights

                    balance_loss_2d = torch.abs(
                        w2d_tensor.narrow(-1, 0, 1) - w2d_tensor.narrow(-1, 1, 1)
                    ).mean()

                    balance_loss_1d = torch.abs(
                        w1d_tensor.narrow(-1, 0, 1) - w1d_tensor.narrow(-1, 1, 1)
                    ).mean()

                    balance_loss = balance_loss_2d + balance_loss_1d
                else:
                    balance_loss = torch.tensor(0.0, device=device)


                if hp.use_xmask:
                    l1_penalty = get_mask_l1_loss(model)
                else:
                    l1_penalty = torch.tensor(0.0, device=device)


                if hp.use_staged_optimization:
                    if epoch > hp.warmup_epochs:
                        if hp.use_dfgu and train_gate:
                            train_loss = train_loss + hp.lambda_balance * balance_loss

                        if hp.use_xmask:
                            progress = min(1.0, (epoch - hp.warmup_epochs) / 10.0)
                            lambda_t = hp.lambda_l1 * progress
                            train_loss = train_loss + lambda_t * l1_penalty
                        else:
                            lambda_t = 0.0
                    else:
                        lambda_t = 0.0
                else:

                    if hp.use_dfgu:
                        train_loss = train_loss + hp.lambda_balance * balance_loss
                    if hp.use_xmask:
                        train_loss = train_loss + hp.lambda_l1 * l1_penalty
                    lambda_t = hp.lambda_l1 if hp.use_xmask else 0.0


                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)

            """valid"""
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for data in valid_loader:

                    data = data.to(device)

                    valid_scores = model(data)

                    valid_labels = data.cls_y
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            avg_w2d = np.mean(epoch_w2d, axis=0)
            avg_w1d = np.mean(epoch_w1d, axis=0)
            print(f"   📊 Modality Weights -> [2D] Drug:{avg_w2d[0]:.3f}, Prot:{avg_w2d[1]:.3f} | [1D] Drug:{avg_w1d[0]:.3f}, Prot:{avg_w1d[1]:.3f}")


            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss_a_epoch:.5f}",
                    f"{valid_loss_a_epoch:.5f}",
                    f"{AUC_dev:.5f}",
                    f"{PRC_dev:.5f}",
                    f"{Accuracy_dev:.5f}",
                    f"{avg_w2d[0]:.3f}",
                    f"{avg_w2d[1]:.3f}"
                ])

            '''save checkpoint and make decision when early stop'''
            early_stopping(AUC_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{str(device).replace(":", "_")}.pth', weights_only=True))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1, IMDF=True)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1, IMDF=True)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1, IMDF=True)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False,
                output_dir=output_root)


def ensemble_run_IMDF_model_legacy(SEED, DATASET, K_Fold, device):
    """Legacy logit-averaging evaluator. Use ensemble_final.py for soft voting."""

    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''Data Preparation'''
    os.makedirs('./DataSets/Preprocessed', exist_ok=True)
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()


    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)


    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

    model = []
    for i in range(K_Fold):
        model.append(IMDFDTI(device=device).to(device))
        '''IMDF-DTI K-Fold train process is necessary'''
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + f'/valid_best_checkpoint-{str(device).replace(":", "_")}.pth', map_location=torch.device(device), weights_only=True))
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'IMDFDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=device, epsilon=hp.loss_epsilon)


    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold, IMDF=True)

    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)


# Existing imports remain usable. The legacy evaluator still averages logits.
run_MIF_model = run_IMDF_model
ensemble_run_MIF_model = ensemble_run_IMDF_model_legacy


