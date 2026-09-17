import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from scipy.sparse import csr_matrix
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from utils import get_dataset, set_train_val_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class Net(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(Net, self).__init__()
        self.fcn1 = nn.Linear(nfeat, nhid)
        self.fcn2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fcn1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn2(x)
        return F.log_softmax(x, dim=1)


class gcn_env(object):
    def __init__(
        self,
        dataset="Cora",
        lr=0.01,
        weight_decay=5e-4,
        max_layer=2,
        batch_size=128,
        policy="",
        data_source="rawr",
        rawr_root=None,
        val_ratio=0.05,
        test_ratio=0.10,
        split_seed=0,
        use_features=True,
        rewiring_mode="none",
        rewiring_epsilon=None,
        partition_path=None,
        partition_edge_path=None,
        partition_id_offset=-1,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_obj = get_dataset(
            dataset,
            data_source=data_source,
            rawr_root=rawr_root,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            use_features=use_features,
            rewiring_mode=rewiring_mode,
            rewiring_epsilon=rewiring_epsilon,
            partition_path=partition_path,
            partition_edge_path=partition_edge_path,
            partition_id_offset=partition_id_offset,
        )
        data = dataset_obj.data

        # Fallback for non-RAwR loaders that do not carry masks.
        if (
            not hasattr(data, "train_mask")
            or data.train_mask is None
            or data.val_mask is None
            or data.test_mask is None
        ):
            n_class = data.y.max() + 1
            n_nodes = data.x.shape[0]
            num_per_class = int((n_nodes * 0.4) / n_class)
            num_development = int(n_nodes * 0.2)
            data = set_train_val_test_split(
                split_seed,
                data,
                num_development,
                num_per_class,
            )

        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        adj = to_dense_adj(data.edge_index).cpu().numpy()[0]
        norm = np.array([np.sum(row) for row in adj])
        norm[norm == 0] = 1.0

        self.adj = (adj / norm).T
        self.max_layer = max(1, int(max_layer))
        self.init_k_hop(self.max_layer)

        n_class = int(data.y.max().item() + 1)
        self.model = Net(data.x.shape[1], 64, n_class).to(device)
        self.data = data.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        train_mask = self.data.train_mask.to("cpu").numpy()
        self.train_indexes = np.where(train_mask == True)[0]
        if len(self.train_indexes) < 2:
            raise ValueError(f"Dataset '{dataset}' has too few training nodes for this environment.")

        self.batch_size = min(batch_size, len(self.train_indexes) - 1)
        self.i = 0
        self.val_acc = 0.0
        self._set_action_space(self.max_layer)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy

        self.baseline_experience = 50
        self.buffers = []
        self.past_performance = [0]
        self.val_acc_dict = [5, 1]

    def agg(self, action, indx):
        edge_index = self.data.edge_index
        row, col = edge_index
        deg = degree(col, self.data.x.size(0), dtype=self.data.x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        feature_list = [self.data.x]
        djmat = torch.sparse.FloatTensor(edge_index, norm)
        # Need up to (hop + 1)-th features due interpolation term.
        for _ in range(1, self.max_layer + 2):
            feature_list.append(torch.spmm(djmat, feature_list[-1]))

        input_feature = []
        for k, node_id in enumerate(indx):
            hop_float = float(action[k][0])
            hop = int(np.clip(round(hop_float), 0, self.max_layer))
            if hop == 0:
                fea = (
                    (hop_float - hop) * feature_list[hop][node_id].unsqueeze(0)
                    + (hop + 1 - hop_float) * feature_list[hop + 1][node_id].unsqueeze(0)
                )
            else:
                fea = 0
                alpha = 0.2
                for j in range(hop):
                    fea += (1 - alpha) * feature_list[j][node_id].unsqueeze(0) + alpha * feature_list[0][
                        node_id
                    ].unsqueeze(0)
                fea = fea / hop
                fea += (
                    (hop_float - hop) * feature_list[hop][node_id].unsqueeze(0)
                    + (hop + 1 - hop_float) * feature_list[hop + 1][node_id].unsqueeze(0)
                )
            input_feature.append(fea)

        return torch.cat(input_feature, dim=0)

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def init_k_hop(self, max_hop):
        sp_adj = csr_matrix(self.adj)
        dd = sp_adj
        self.adjs = [dd]
        for _ in range(max_hop):
            dd *= sp_adj
            self.adjs.append(dd)

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.data.x[index].to("cpu").numpy()
        self.optimizer.zero_grad()
        return state

    def _set_action_space(self, _max):
        # One scalar action per node; values are clipped by `self.max_layer`.
        self.action_num = 1

    def action_space(self, s, action_num):
        action = np.random.normal(2, 1, (s, action_num))
        return action

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float("inf"))
        high = np.full(observation.shape, float("inf"))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset2(self):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        state = self.data.x[index].to("cpu").numpy()
        self.optimizer.zero_grad()
        return state

    def step2(self, actions):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]

        actions = np.asarray(actions).reshape(-1, 1)
        clipped_actions = np.clip(actions, 0, self.max_layer)

        feature = self.agg(clipped_actions, index)
        self.train(feature, index)

        done = True

        index = self.stochastic_k_hop(clipped_actions, index)
        next_state = self.data.x[index].to("cpu").numpy()

        val_acc = self.eval_batch()
        self.val_acc_dict.append(val_acc)
        reward = np.mean(val_acc - np.array(self.val_acc_dict))

        return next_state, reward, [done] * self.batch_size, val_acc

    def stochastic_k_hop(self, actions, index):
        next_batch = []
        for idx, act in zip(index, actions):
            hop = int(np.clip(round(float(act[0])), 0, self.max_layer))
            prob = self.adjs[hop].getrow(idx).toarray().flatten()
            if prob.sum() <= 0:
                next_batch.append(idx)
                continue
            prob /= prob.sum()
            cand = np.arange(len(prob))
            next_cand = np.random.choice(cand, p=prob)
            next_batch.append(next_cand)
        return next_batch

    def train(self, feature, indexes):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(feature)
        y = self.data.y[indexes]
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

    def eval_batch(self):
        self.model.eval()
        val_index = np.where(self.data.val_mask.to("cpu").numpy() == True)[0]
        val_states = self.data.x[val_index].to("cpu").numpy()

        val_acts = self.policy.select_action(val_states).clip(0, self.max_layer)
        val_acts = val_acts.reshape(-1, 1)

        feature = self.agg(val_acts, val_index)
        logits = self.model(feature)
        pred = logits.max(1)[1]
        acc = pred.eq(self.data.y[val_index]).sum().item() / len(val_index)
        return acc

    def test_batch(self):
        self.model.eval()
        test_index = np.where(self.data.test_mask.to("cpu").numpy() == True)[0]
        test_states = self.data.x[test_index].to("cpu").numpy()

        test_acts = self.policy.select_action(test_states).clip(0, self.max_layer)
        test_acts = test_acts.reshape(-1, 1)
        feature = self.agg(test_acts, test_index)
        logits = self.model(feature)
        pred = logits.max(1)[1]
        acc = pred.eq(self.data.y[test_index]).sum().item() / len(test_index)
        return acc
