import numpy as np
from multiprocessing import Process, Queue
from scipy.sparse import lil_matrix


def sample_function(user_item_matrix, batch_size, n_negative, result_queue, check_negative=True):
    user_item_matrix = lil_matrix(user_item_matrix)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
    user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}

    while True:
        np.random.shuffle(user_item_pairs)
        for i in range(len(user_item_pairs) // batch_size):
            batch_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]

            # Отрицательные примеры
            negative_samples = np.random.randint(
                0, user_item_matrix.shape[1],
                size=(batch_size, n_negative)
            )

            if check_negative:
                for idx, (user_positive, negatives) in enumerate(zip(batch_pairs, negative_samples)):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        while neg in user_to_positive_set[user]:
                            neg = np.random.randint(0, user_item_matrix.shape[1])
                            negative_samples[idx, j] = neg

            result_queue.put((batch_pairs, negative_samples))


class WarpSampler:
    #(пользователь, положительный айтем) + отрицательные айтемы
    def __init__(self, user_item_matrix, batch_size=10000, n_negative=10, n_workers=5, check_negative=True):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []

        for _ in range(n_workers):
            p = Process(target=sample_function,
                        args=(user_item_matrix, batch_size, n_negative, self.result_queue, check_negative))
            p.start()
            self.processors.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
