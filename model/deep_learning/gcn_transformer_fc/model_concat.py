import csv
import os
import pickle
from pathlib import Path

import torch.nn as nn
from models import Transformer_test, Model_TGCN, batch_size
from data_pretreatment import func
# Qualified import on purpose: `from dgllife.utils import *` below also exports an
# `EarlyStopping` (different signature, no min_delta). A bare `from early_stopping
# import EarlyStopping` here would be silently clobbered by that wildcard import.
import early_stopping
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import dgl
from dgllife.utils import *
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
torch.cuda.empty_cache()

RESULTS_CSV = Path(__file__).parent / "results_per_fold.csv"

# Early stopping: halt a fold once validation loss has not improved for
# EARLY_STOP_PATIENCE consecutive epochs. The model kept is the one at the stop
# epoch (no best-weight restore). The 500-epoch loop below stays as a safety
# ceiling that early stopping normally fires well before.
EARLY_STOP_PATIENCE = 50
EARLY_STOP_MIN_DELTA = 0.0

def main_():
    for num in range(1, 11):
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(num)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(num)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(num)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_trans = Transformer_test().to(device)
        torch.cuda.empty_cache()
        df_seq_train, y_train_tensor, y_true_train, list_num_train = func(PATH_x_train)
        df_seq_test, y_test_tensor, y_true_test, list_num_test = func(PATH_x_test)
        df_seq_val, y_val_tensor, y_true_val, list_num_val = func(PATH_x_val)

        device = 'cpu'
        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
        mol = Chem.MolFromSmiles('c1ccccc1')
        n_feats = atom_featurizer.feat_size('feat')
        print(n_feats)

        def get_data(df):
            mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
            g = [mol_to_complete_graph(m, node_featurizer=node_featurizer) for m in mols]
            y = np.array(list((df['label'])))
            y = np.array(y, dtype=np.int64)
            return g, y

        gcn_net = GCNPredictor(in_feats=n_feats,
                               hidden_feats=[60, 20],
                               n_tasks=2,
                               predictor_hidden_feats=10,
                               predictor_dropout=0.5, )
        gcn_net = gcn_net.to(device)
        model_tgcn = Model_TGCN().to(device)

        def collate(sample):
            _, list_num, graphs, labels, index = map(list, zip(*sample))
            batched_graph = dgl.batch(graphs)
            batched_graph.set_n_initializer(dgl.init.zero_initializer)
            batched_graph.set_e_initializer(dgl.init.zero_initializer)
            return _, list_num, batched_graph, torch.tensor(labels), index

        train_X = pd.read_csv(PATH_x_train)
        x_train, y_train = get_data(train_X)
        train_data = list(zip(df_seq_train, list_num_train, x_train, y_train, [i for i in range(len(train_X))]))

        # 87/13 class imbalance: oversample minority class per batch via
        # WeightedRandomSampler. Weights are 1 / class_count, so each draw is
        # uniform over classes rather than uniform over samples. replacement=True
        # is required when num_samples >= len(weights).
        y_train_int = np.asarray(y_train).astype(int).ravel()
        class_counts = np.bincount(y_train_int, minlength=2)
        weights_per_class = 1.0 / np.maximum(class_counts, 1)
        sample_weights = torch.tensor(
            [weights_per_class[y] for y in y_train_int], dtype=torch.float
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader_ = DataLoader(train_data, batch_size=batch_size, sampler=sampler, collate_fn=collate, drop_last=True)

        test_X = pd.read_csv(PATH_x_test)
        x_test, y_test = get_data(test_X)
        test_data = list(zip(df_seq_test, list_num_test, x_test, y_test, [i for i in range(len(test_X))]))
        test_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate,
                                      drop_last=True)

        val_X = pd.read_csv(PATH_x_val)
        x_val, y_val = get_data(val_X)
        val_data = list(zip(df_seq_val, list_num_val, x_val, y_val, [i for i in range(len(val_X))]))
        val_loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)


        optimizer = torch.optim.Adam([{'params': gcn_net.parameters()},
                                      {'params': model_trans.parameters()},
                                      {'params': model_tgcn.parameters()}], lr=0.001)

        stopper = early_stopping.EarlyStopping(patience=EARLY_STOP_PATIENCE,
                                               min_delta=EARLY_STOP_MIN_DELTA, mode="min")

        for epoch in range(1, 501):
            # train
            gcn_net.train()
            model_trans.train()
            model_tgcn.train()
            train_epoch_loss, train_epoch_acc, train_epoch_r2 = 0, 0, 0
            for i, (X, list_num, graph, labels, index) in enumerate(train_loader_):
                train_labels = labels.to(device)
                atom_feats = graph.ndata.pop('h').to(device)
                atom_feats, train_labels = atom_feats.to(device), train_labels.to(device)
                # Using GCN to obtain GCN encoding for sequence information
                train_pred = gcn_net(graph, atom_feats, model_use='a')
                X = torch.cat(X, dim=0)
                X = torch.reshape(X, [batch_size, 128])
                X = X.to("cuda")
                list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num])
                y = model_trans(X)
                y = model_tgcn(y, train_pred, list_num).to("cpu")
                y = torch.reshape(y, [batch_size])
                train_loss = nn.BCELoss()(y, train_labels.float())
                optimizer.zero_grad()
                train_loss.requires_grad_(True)
                train_loss.backward()
                optimizer.step()
                train_epoch_loss += train_loss.detach().item()
                trlist_numain_pred_cls = train_pred.argmax(-1).detach().to('cpu').numpy()
                train_true_label = train_labels.to('cpu').numpy()
                yy = [1 if i >= 0.5 else 0 for i in y.detach().numpy()]
                train_epoch_acc += sum(train_true_label == yy)

            train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
            train_epoch_acc /= (i + 1)
            train_epoch_loss /= (i + 1)


            # Per-epoch checkpoint saves disabled: with 500 epochs x 10 folds x
            # 3 models = 15k file writes, the upstream behavior fills ~90 GB
            # (and previously much more, see tsne_list note in models.py). We
            # only need per-epoch predictions for the comparison; those still
            # land in pred_data_origin/ as small CSVs. If you need checkpoints
            # for analysis later, re-enable just for the final epoch:
            #   if epoch == 500:
            #       os.makedirs(f'./model_origin/gcn_transformer_fc/{num}/', exist_ok=True)
            #       torch.save(model_tgcn, f'./model_origin/gcn_transformer_fc/{num}/final_tgcn.pt')

            def train_test_val(dataloader):
                epoch_loss, epoch_acc = 0, 0
                mlist = []
                gcn_net.eval()
                model_trans.eval()
                model_tgcn.eval()
                for i, (X, list_num, graph, labels, index) in enumerate(dataloader):
                    labels = labels.to(device)
                    atom_feats = graph.ndata.pop('h').to(device)
                    atom_feats, labels = atom_feats.to(device), labels.to(device)
                    # Using GCN to obtain GCN encoding for sequence information
                    pred = gcn_net(graph, atom_feats, model_use='a')
                    X = torch.cat(X, dim=0)
                    X = torch.reshape(X, [batch_size, 128])
                    X = X.to("cuda")
                    list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num]).cuda()
                    y = model_trans(X)
                    y = model_tgcn(y, pred, list_num).to("cpu")
                    y = torch.reshape(y, [batch_size])
                    loss = nn.BCELoss()(y, labels.float())
                    epoch_loss += loss.detach().item()
                    pred_cls = y.detach().numpy()
                    true_label = labels.to('cpu').numpy()
                    yy = [1 if m >= 0.5 else 0 for m in y.detach().numpy()]
                    mlist.extend(pred_cls)
                    epoch_acc += sum(true_label == yy)
                epoch_acc = epoch_acc / true_label.shape[0]
                epoch_acc /= (i + 1)
                epoch_loss /= (i + 1)

                return epoch_acc, epoch_loss, mlist

            test_epoch_acc, test_epoch_loss, test_list = train_test_val(test_loader_test)
            val_epoch_acc, val_epoch_loss, val_list = train_test_val(val_loader_val)

            print(f"epoch: {epoch}, train_LOSS      : {train_epoch_loss:.3f}, train_ACC        : {train_epoch_acc:.3f}")
            print(f"epoch: {epoch}, test_LOSS       : {test_epoch_loss:.3f}, test_ACC         : {test_epoch_acc:.3f}")
            print(f"epoch: {epoch}, val_LOSS        : {val_epoch_loss:.3f}, val_ACC          : {val_epoch_acc:.3f}")

            y_true_test = pd.read_csv(PATH_x_test, usecols=['label']).values
            t1, t2 = pd.DataFrame(test_list, columns=['predict']), pd.DataFrame(y_true_test, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/gcn_transformer_fc/{num}/test/', exist_ok=True)
            pd.DataFrame(tt).to_csv(
                './pred_data_origin/gcn_transformer_fc/{}/test/experiment_{}_predicted_test_values.csv'.format(num, epoch),
                index=False)
            y_true_val = pd.read_csv(PATH_x_val, usecols=['label']).values
            t1, t2 = pd.DataFrame(val_list, columns=['predict']), pd.DataFrame(y_true_val, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/gcn_transformer_fc/{num}/val/', exist_ok=True)
            pd.DataFrame(tt).to_csv(
                './pred_data_origin/gcn_transformer_fc/{}/val/experiment_{}_predicted_valid_values.csv'.format(num, epoch),
                index=False)

            # Early stopping on validation loss. step() is called once per epoch;
            # it returns True once val loss has not improved for EARLY_STOP_PATIENCE
            # epochs. We keep the current (stop-epoch) weights -- the post-loop
            # scoring block below already uses this epoch's test_list/val_list.
            if stopper.step(val_epoch_loss):
                print(
                    f"Fold {num}: early stopping at epoch {epoch} "
                    f"(best val loss {stopper.best:.4f} @ epoch {stopper.best_epoch}; "
                    f"no improvement for {EARLY_STOP_PATIENCE} epochs)"
                )
                break

        # Final-epoch test/val metrics. drop_last=True in the loaders truncates
        # the eval sets, so len(test_list) <= len(PATH_x_test rows); we score
        # only the matched prefix and record both counts for transparency.
        y_test_full = pd.read_csv(PATH_x_test, usecols=['label']).values.ravel().astype(int)
        y_val_full = pd.read_csv(PATH_x_val, usecols=['label']).values.ravel().astype(int)
        n_test, n_val = len(test_list), len(val_list)
        y_test_match = y_test_full[:n_test]
        y_val_match = y_val_full[:n_val]
        test_preds = np.asarray(test_list).ravel()
        val_preds = np.asarray(val_list).ravel()
        test_bin = (test_preds >= 0.5).astype(int)
        val_bin = (val_preds >= 0.5).astype(int)
        test_auc = (
            float(roc_auc_score(y_test_match, test_preds))
            if len(set(y_test_match.tolist())) > 1 else float('nan')
        )
        val_auc = (
            float(roc_auc_score(y_val_match, val_preds))
            if len(set(y_val_match.tolist())) > 1 else float('nan')
        )
        test_f1 = float(f1_score(y_test_match, test_bin, zero_division=0))
        val_f1 = float(f1_score(y_val_match, val_bin, zero_division=0))
        test_acc = float(accuracy_score(y_test_match, test_bin))
        val_acc = float(accuracy_score(y_val_match, val_bin))
        print(
            f"Fold {num} final (epoch {epoch}): "
            f"test AUC={test_auc:.4f} F1={test_f1:.4f} ACC={test_acc:.4f} "
            f"on {n_test}/{len(y_test_full)} samples"
        )
        write_header = not RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow([
                    "fold", "epoch",
                    "test_auc", "test_f1", "test_accuracy",
                    "val_auc", "val_f1", "val_accuracy",
                    "n_test_eval", "n_test_total",
                    "n_val_eval", "n_val_total",
                    "best_val_loss_epoch", "best_val_loss",
                ])
            writer.writerow([
                num, epoch,
                test_auc, test_f1, test_acc,
                val_auc, val_f1, val_acc,
                n_test, len(y_test_full),
                n_val, len(y_val_full),
                stopper.best_epoch, stopper.best,
            ])


if __name__ == '__main__':
    main_()

