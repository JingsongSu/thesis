import math


def normalize_code_text(x):
    return str(x).strip().replace(" ", "")


def get_topk_results(predictions, scores, targets, k, all_items=None):
    """
    predictions: List[str], length = B * k
    scores: List[float] or Tensor, length = B * k
    targets: List[str], length = B
    return: List[List[int]]  shape = [B, k]

    当前版本默认：
    - predictions 只包含 fine code 文本
    - targets 只包含 fine code 文本
    """
    results = []
    B = len(targets)

    predictions = [normalize_code_text(p) for p in predictions]
    scores = [float(s) for s in scores]
    targets = [normalize_code_text(t) for t in targets]

    # 非法 item 直接置极小分
    if all_items is not None:
        all_items = set([normalize_code_text(x) for x in all_items])
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1e9

    for b in range(B):
        batch_seqs = predictions[b * k:(b + 1) * k]
        batch_scores = scores[b * k:(b + 1) * k]

        # 加入 beam_id 作为稳定 tie-breaker
        pairs = [
            (beam_id, seq, score)
            for beam_id, (seq, score) in enumerate(zip(batch_seqs, batch_scores))
        ]

        # score 高的优先；若 score 相同，beam_id 小的优先
        pairs.sort(key=lambda x: (x[2], -x[0]), reverse=True)

        target_item = targets[b]
        one_results = [1 if seq == target_item else 0 for _, seq, _ in pairs]
        results.append(one_results)

    return results


def get_metrics_results(topk_results, metrics):
    """
    返回“分子和”，不是均值
    可直接用于 DDP all_reduce
    """
    res = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower.startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m_lower.startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"Unknown metric: {m}")
    return res


def ndcg_k(topk_results, k):
    """
    Leave-one-out 场景：
    - 每个 sample 只有一个正样本
    - IDCG = 1
    返回 DCG 的“分子和”，不是均值
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        for i, v in enumerate(res):
            if v > 0:
                ndcg += 1.0 / math.log2(i + 2)
                break
    return ndcg


def hit_k(topk_results, k):
    """
    Hit@K：命中计数
    """
    hit = 0.0
    for row in topk_results:
        if sum(row[:k]) > 0:
            hit += 1.0
    return hit
