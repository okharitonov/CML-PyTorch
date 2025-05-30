import torch
import numpy as np
from scipy.sparse import lil_matrix


class RecallEvaluator:
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix, device='cpu'):
        self.device = device
        self.model = model.to(device)

        self.train_mat = lil_matrix(train_user_item_matrix)
        self.test_mat = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]

        self.user_to_train_set = {
            u: set(self.train_mat.rows[u])
            for u in range(n_users)
            if self.train_mat.rows[u]
        }
        self.user_to_test_set = {
            u: set(self.test_mat.rows[u])
            for u in range(n_users)
            if self.test_mat.rows[u]
        }

        self.max_train_count = max(
            (len(rows) for rows in self.train_mat.rows), default=0
        )

    def eval(self, users, k=50, batch_size=100):
        """
        Recall@k для списка пользователей.
        :param users: итерируемый объект с user_id (int)
        :param k: топ-K
        :param batch_size: размер батча при поиске top-k
        :return: список recall для каждого пользователя
        """
        recalls = []
        users = list(users)

        topn = k + self.max_train_count

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(users), batch_size):
                batch_users = users[i : i + batch_size]
                u_tensor = torch.tensor(batch_users, dtype=torch.long, device=self.device)

                scores = self.model.score(u_tensor)

                top_indices = torch.topk(scores, topn, dim=1).indices.cpu().numpy()

                for idx, u in enumerate(batch_users):
                    train_set = self.user_to_train_set.get(u, set())
                    test_set = self.user_to_test_set.get(u, set())

                    if not test_set:
                        recalls.append(0.0)
                        continue

                    hits = 0
                    shown = 0
                    for item_id in top_indices[idx]:
                        if item_id in train_set:
                            continue
                        shown += 1
                        if item_id in test_set:
                            hits += 1
                        if shown == k:
                            break

                    recalls.append(hits / len(test_set))

        return recalls
