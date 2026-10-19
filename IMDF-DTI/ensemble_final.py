import torch
import numpy as np
import os
import joblib
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score)
import torch_geometric.loader as pyg_loader
from datetime import datetime
from pathlib import Path
import argparse
import json
from checkpoints import find_checkpoint


from model import IMDFDTI
from config import hyperparameter
from utils.DataPrepare import shuffle_dataset
from utils import ProteinMoleculeDataset

def run_ensemble(dataset="BIOSNAP", seed=114514, gpu=0, folds=5,
                 checkpoint_dir=None, output_dir=None):
    hp = hyperparameter()


    SEED = seed
    DATASET = dataset
    if DATASET not in ["DrugBank", "BIOSNAP", "Davis"]:
        raise ValueError("Expected DrugBank, BIOSNAP, or Davis")
    if folds < 2:
        raise ValueError("At least two fold-trained models are required")
    DEVICE = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    DEV_STR = str(DEVICE).replace(":", "_")
    checkpoint_root = Path(checkpoint_dir or DATASET)
    # Fail before expensive loading/inference if any fold is missing or ambiguous.
    checkpoint_paths = [find_checkpoint(checkpoint_root / str(i)) for i in range(1, folds + 1)]


    print(f"⭐ IMDF-DTI 深度集成系统 | 数据集: {DATASET} | 设备: {DEVICE}")


    print("步骤 [1/7]: 正在同步数据切分逻辑...")
    dir_input = f'./DataSets/{DATASET}.txt'
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')

    data_list = shuffle_dataset(data_list, SEED)
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print(f"✅ 数据对齐完成，独立测试集样本数: {len(test_data_list)}")


    print("步骤 [2/7]: 正在加载预处理字典...")
    try:
        protein_path = f'./DataSets/Preprocessed/{DATASET}-protein-new.pkl'
        ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi-new.pkl'
        protein_dict = joblib.load(protein_path)
        ligand_dict = joblib.load(ligand_path)
        print("   - 字典文件已成功载入内存")
    except Exception as e:
        raise RuntimeError("Feature-cache loading failed; run prepare_data.py first.") from e


    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=DEVICE)
    test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False,
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])


    print(f"步骤 [3/7]: 正在载入 {folds} 折模型权重...")
    models = []
    all_mask_weights = {'mask_2d_drug': [], 'mask_2d_prot': [], 'mask_1d_drug': [], 'mask_1d_prot': []}


    batch_dfgu_2d = []
    batch_dfgu_1d = []

    for i in range(1, folds + 1):
        m = IMDFDTI(depth=3, device=DEVICE).to(DEVICE)
        path = checkpoint_paths[i - 1]
        metadata_path = checkpoint_root / str(i) / 'hyperparameters.json'
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if any(metadata.get(k, True) is False for k in
                   ['use_xmask', 'use_dfgu', 'use_rescal_hybrid', 'use_staged_optimization']):
                raise ValueError(f"{metadata_path} describes an ablation, not a full-model ensemble.")

        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.eval()
            models.append(m)


            exp_w = m.get_explainability_weights()
            for k in all_mask_weights.keys():
                all_mask_weights[k].append(exp_w[k])
            print(f"   [Fold {i}] 权重载入成功")
        else:
            raise FileNotFoundError(path)

    if not models:
        print("❌ 错误：未加载到任何权重，程序终止。")
        return


    print(f"步骤 [4/7]: 正在执行多模型联合表决 (Soft Voting)...")
    all_labels, all_probs = [], []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="推理进度"):
            data = data.to(DEVICE)
            model_probs = []

            for m in models:
                outputs = m(data)
                probs = F.softmax(outputs, dim=1)[:, 1]
                model_probs.append(probs.cpu().numpy())


                if hasattr(m.dfgu_2d, 'latest_weights'):
                    batch_dfgu_2d.append(m.dfgu_2d.latest_weights.cpu().numpy())
                    batch_dfgu_1d.append(m.dfgu_1d.latest_weights.cpu().numpy())

            ensemble_prob = np.mean(model_probs, axis=0)


            all_probs.extend(ensemble_prob)
            all_labels.extend(data.cls_y.cpu().numpy())


    print("步骤 [5/7]: 正在结算全量集成指标...")
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.50).astype(int)

    final_auc = roc_auc_score(y_true, y_prob)
    final_prc = average_precision_score(y_true, y_prob)
    final_acc = accuracy_score(y_true, y_pred)
    final_prec = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)


    print("步骤 [6/7]: 正在保存结果文件...")
    save_dir = str(output_dir or checkpoint_root)
    if not os.path.exists(save_dir): os.makedirs(save_dir)


    res_path = os.path.join(save_dir, "ensemble_results.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(res_path, "a") as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"数据集: {DATASET} | 随机种子: {SEED}\n")
        f.write(f"ROC-AUC  : {final_auc:.5f}\n")
        f.write(f"PRC/AUPR : {final_prc:.5f}\n")
        f.write(f"Accuracy : {final_acc:.5f}\n")
        f.write(f"Precision: {final_prec:.5f}\n")
        f.write(f"Recall   : {final_recall:.5f}\n")
        f.write(f"{'='*40}\n")


    np.save(os.path.join(save_dir, "ensemble_y_prob.npy"), y_prob)
    np.save(os.path.join(save_dir, "ensemble_y_true.npy"), y_true)


    print("步骤 [7/7]: 正在导出集成平均解释性权重...")
    ensemble_explain = {
        'mask_2d_drug': np.mean(all_mask_weights['mask_2d_drug'], axis=0),
        'mask_2d_prot': np.mean(all_mask_weights['mask_2d_prot'], axis=0),
        'mask_1d_drug': np.mean(all_mask_weights['mask_1d_drug'], axis=0),
        'mask_1d_prot': np.mean(all_mask_weights['mask_1d_prot'], axis=0),
        'dfgu_2d_mean': np.mean(batch_dfgu_2d, axis=0) if batch_dfgu_2d else None,
        'dfgu_1d_mean': np.mean(batch_dfgu_1d, axis=0) if batch_dfgu_1d else None
    }
    np.save(os.path.join(save_dir, "ensemble_weights.npy"), ensemble_explain)


    print("\n" + "🏆" * 20)
    print(f"🔥 {DATASET} 最终集成战报 🔥")
    print(f"   ROC-AUC  : {final_auc:.5f}")
    print(f"   PRC/AUPR : {final_prc:.5f}")
    print(f"   Accuracy : {final_acc:.5f}")
    print(f"   Precision: {final_prec:.5f}")
    print(f"   Recall   : {final_recall:.5f}")
    print(f"✅ 所有结果、原始预测数据及集成权重已保存至: {save_dir}")
    print("🏆" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDF-DTI-B equal-weight probability averaging.")
    parser.add_argument('dataset', choices=['DrugBank', 'BIOSNAP', 'Davis'])
    parser.add_argument('-s', '--seed', type=int, default=114514)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-f', '--folds', type=int, default=5)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run_ensemble(args.dataset, args.seed, args.gpu, args.folds,
                 args.checkpoint_dir, args.output_dir)
