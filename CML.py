import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from evaluator import RecallEvaluator
from sampler import WarpSampler
from utils import citeulike, movielens, split_data
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

class CML(nn.Module):
    def __init__(self, n_users, n_items,
                 embed_dim=100,
                 features=None,
                 margin=0.5,
                 hidden_layer_dim=256,
                 dropout_rate=0.5,
                 feature_l2_reg=0.5,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=1):
        super(CML, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.margin = margin
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor

        # Embeddings
        self.user_embeddings = nn.Embedding(n_users, embed_dim)
        self.item_embeddings = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embeddings.weight, std=1.0 / (embed_dim ** 0.5))
        nn.init.normal_(self.item_embeddings.weight, std=1.0 / (embed_dim ** 0.5))

        if features is not None:
            feat_dim = features.shape[1]
            self.register_buffer('features', torch.tensor(features, dtype=torch.float32))
            self.mlp_layer_1 = nn.Linear(feat_dim, hidden_layer_dim)
            self.dropout = nn.Dropout(dropout_rate)
            self.mlp_layer_2 = nn.Linear(hidden_layer_dim, embed_dim)
        else:
            self.features = None

    def feature_projection(self):
        if self.features is None:
            return None
        x = F.relu(self.mlp_layer_1(self.features))
        x = self.dropout(x)
        proj = self.mlp_layer_2(x) * self.feature_projection_scaling_factor
        return F.normalize(proj, p=2, dim=1)

    def covariance_loss(self):
        X = torch.cat([self.item_embeddings.weight, self.user_embeddings.weight], dim=0)
        X = X - X.mean(dim=0, keepdim=True)
        cov = (X.t() @ X) / X.size(0)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.abs().sum() * self.cov_loss_weight

    def embedding_loss(self, user_ids, pos_item_ids, neg_item_ids):
        u = self.user_embeddings(user_ids)
        pos = self.item_embeddings(pos_item_ids)
        pos_dist = torch.sum((u - pos)**2, dim=1)

        neg = self.item_embeddings(neg_item_ids)
        u_exp = u.unsqueeze(1)                          
        neg_dist = torch.sum((u_exp - neg)**2, dim=2)

        closest_neg, _ = torch.min(neg_dist, dim=1)
        loss_per_pair = F.relu(pos_dist - closest_neg + self.margin)

        if self.use_rank_weight:
            impostors = (pos_dist.unsqueeze(1) - neg_dist + self.margin) > 0
            rank = impostors.float().mean(dim=1) * self.n_items
            loss_per_pair = loss_per_pair * torch.log(rank + 1)

        return loss_per_pair.sum()

    def feature_loss(self):
        proj = self.feature_projection()
        if proj is None:
            return torch.tensor(0.0, device=self.user_embeddings.weight.device)
        diff = self.item_embeddings.weight - proj
        return (diff.pow(2).sum()) * self.feature_l2_reg

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        loss = self.embedding_loss(user_ids, pos_item_ids, neg_item_ids)
        loss = loss + self.feature_loss()
        if self.use_cov_loss:
            loss = loss + self.covariance_loss()
        return loss

    def score(self, user_ids):
        u = self.user_embeddings(user_ids)
        all_items = self.item_embeddings.weight.unsqueeze(0)
        u_exp = u.unsqueeze(1)
        dist = torch.sum((u_exp - all_items)**2, dim=2)
        return -dist

BATCH_SIZE = 50000
N_NEGATIVE = 20
EVALUATION_EVERY_N_BATCHES = 50
EMBED_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize(model, sampler, train_matrix, valid_matrix, n_users, n_items, use_scheduler=None):
    model.to(DEVICE)
    #Choose Optimizer
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

    scheduler = None
    if use_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    elif use_scheduler == 'linear':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1 - step/1000)

    
    valid_users = np.random.choice(list(set(valid_matrix.nonzero()[0])), size=1000, replace=False)
    evaluator = RecallEvaluator(model, train_matrix, valid_matrix, device=DEVICE)

    step_count = 0
    
    while True:
        recalls = evaluator.eval(valid_users, k=50)
        print(f"\nRecall@50 on validation set: {np.mean(recalls):.4f}")

        losses = []
        for _ in tqdm(range(EVALUATION_EVERY_N_BATCHES), desc="Training"):
            user_pos_pairs, neg_items = sampler.next_batch()
            user_ids = torch.tensor(user_pos_pairs[:, 0], dtype=torch.long, device=DEVICE)
            pos_item_ids = torch.tensor(user_pos_pairs[:, 1], dtype=torch.long, device=DEVICE)
            neg_item_ids = torch.tensor(neg_items, dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            loss = model(user_ids, pos_item_ids, neg_item_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()  
                step_count += 1

                if step_count % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"[LR: {current_lr:.2e}]", end=" ")
            
            losses.append(loss.item())

        print(f"\nAverage training loss: {np.mean(losses):.4f}")
#Loading Data
user_item_matrix, features = movielens(
    "data/sample_interactions.csv",
    feature_h5_path="data/sample_overview_vectors.h5" 
)
        
n_users, n_items = user_item_matrix.shape
dense_features = features.toarray().astype(np.float32) if features.nnz > 0 else None
if dense_features is not None:
    dense_features += 1e-10
train, valid, test = split_data(user_item_matrix)


sampler = WarpSampler(train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)

model = CML(n_users=n_users,
            n_items=n_items,
            embed_dim=EMBED_DIM,
            features=dense_features,
            margin=0.5,
            hidden_layer_dim=256,
            dropout_rate=0.5,
            feature_projection_scaling_factor=1.0,
            feature_l2_reg=0.5,
            use_rank_weight=True,
            use_cov_loss=False,
            cov_loss_weight=1.0)

    #use_scheduler: None/'cosine'/'linear'
optimize(model, sampler, train, valid, n_users, n_items, use_scheduler=None)