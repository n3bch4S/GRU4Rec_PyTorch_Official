from gru4rec_pytorch import SessionDataIterator
import torch


@torch.no_grad()
def batch_eval(
    gru,
    test_data,
    cutoff=[20],
    batch_size=512,
    mode="conservative",
    item_key="ItemId",
    session_key="SessionId",
    time_key="Time",
):
    if gru.error_during_train:
        raise Exception(
            "Attempting to evaluate a model that wasn't trained properly (error_during_train=True)"
        )

    recall = dict()
    mrr = dict()
    hr_at_k = dict()
    ndcg_at_k = dict()

    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
        hr_at_k[c] = 0
        ndcg_at_k[c] = 0

    H = []
    for i in range(len(gru.layers)):
        H.append(
            torch.zeros(
                (batch_size, gru.layers[i]),
                requires_grad=False,
                device=gru.device,
                dtype=torch.float32,
            )
        )

    n = 0
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(
        n_valid, finished_mask, valid_mask, H
    )
    data_iterator = SessionDataIterator(
        test_data,
        batch_size,
        0,
        0,
        0,
        item_key,
        session_key,
        time_key,
        device=gru.device,
        itemidmap=gru.data_iterator.itemidmap,
    )

    for in_idxs, out_idxs in data_iterator(
        enable_neg_samples=False, reset_hook=reset_hook
    ):
        for h in H:
            h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])

        if mode == "standard":
            ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == "conservative":
            ranks = (oscores >= tscores).sum(dim=0)
        elif mode == "median":
            ranks = (
                (oscores > tscores).sum(dim=0)
                + 0.5 * ((oscores == tscores).dim(axis=0) - 1)
                + 1
            )
        else:
            raise NotImplementedError

        for c in cutoff:
            recall[c] += (ranks <= c).sum().cpu().numpy()  # Calculate recall@k
            mrr[c] += (
                ((ranks <= c) / ranks.float()).sum().cpu().numpy()
            )  # Calculate MRR@k

            # Calculate HR@k: Hit Ratio at k (whether the true item is in top-k)
            hr_at_k[c] += (
                (
                    (ranks <= c)
                    & (
                        out_idxs.unsqueeze(0)
                        == torch.arange(len(ranks), device=gru.device).unsqueeze(1)
                    ).sum(dim=0)
                    > 0
                )
                .sum()
                .cpu()
                .numpy()
            )

            # Calculate NDCG@k: Normalized Discounted Cumulative Gain at k
            ideal_dcg = 1.0 / torch.log2(
                torch.arange(2, c + 2, device=gru.device).float()
            )  # Ideal DCG
            dcg = (1.0 / torch.log2(ranks.float() + 1)).sum(
                dim=0
            )  # DCG for this session
            idcg = ideal_dcg.sum()  # Ideal DCG (IDCG)
            ndcg_at_k[c] += (
                (dcg / idcg).sum().cpu().numpy() if idcg > 0 else 0
            )  # NDCG@k

        n += O.shape[0]

    # Average over the total number of sessions
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n
        hr_at_k[c] /= n
        ndcg_at_k[c] /= n

    return recall, mrr, hr_at_k, ndcg_at_k
