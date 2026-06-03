
import math


def _normalize_text(x):
    if x is None:
        return None
    x = str(x)
    x = x.split("Response:")[-1]
    return x.strip().replace(" ", "")


def get_topk_results(predictions, scores, fine_targets, coarse_targets, k, all_items=None):
    """
    返回:
      results: List[List[int]]
      top1_pairs: List[dict]
        {
          "pred": ...,
          "fine_target": ...,
          "coarse_target": ...
        }
    """
    results = []
    top1_pairs = []

    B = len(fine_targets)
    predictions = [_normalize_text(_) for _ in predictions]
    fine_targets = [_normalize_text(_) for _ in fine_targets]
    coarse_targets = [_normalize_text(_) for _ in coarse_targets]

    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().tolist()
    else:
        scores = list(scores)

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, s) for a, s in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        fine_target = fine_targets[b]
        coarse_target = coarse_targets[b]

        valid_targets = {fine_target}
        if coarse_target is not None:
            valid_targets.add(coarse_target)

        top1_pred = sorted_pairs[0][0]
        top1_pairs.append({
            "pred": top1_pred,
            "fine_target": fine_target,
            "coarse_target": coarse_target,
        })

        one_results = []
        for pred, _ in sorted_pairs:
            one_results.append(1 if pred in valid_targets else 0)

        results.append(one_results)

    return results, top1_pairs


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError
    return res


def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        for i, v in enumerate(res):
            if v > 0:
                ndcg += 1.0 / math.log(i + 2, 2)
                break
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit
