# import math

# def get_topk_results(predictions, scores, targets, k, all_items=None):
#     results = []
#     B = len(targets)
#     predictions = [_.split("Response:")[-1] for _ in predictions]
#     predictions = [_.strip().replace(" ","") for _ in predictions]

#     if all_items is not None:
#         for i, seq in enumerate(predictions):
#             if seq not in all_items:
#                 scores[i] = -1000

#     for b in range(B):
#         batch_seqs = predictions[b * k: (b + 1) * k]
#         batch_scores = scores[b * k: (b + 1) * k]

#         pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
#         sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
#         target_item = targets[b]
#         one_results = []
#         for sorted_pred in sorted_pairs:
#             if sorted_pred[0] == target_item:
#                 one_results.append(1)
#             else:
#                 one_results.append(0)

#         results.append(one_results)

#     return results

# def get_metrics_results(topk_results, metrics):
#     res = {}
#     for m in metrics:
#         if m.lower().startswith("hit"):
#             k = int(m.split("@")[1])
#             res[m] = hit_k(topk_results, k)
#         elif m.lower().startswith("ndcg"):
#             k = int(m.split("@")[1])
#             res[m] = ndcg_k(topk_results, k)
#         else:
#             raise NotImplementedError

#     return res


# def ndcg_k(topk_results, k):

#     ndcg = 0.0
#     for row in topk_results:
#         res = row[:k]
#         one_ndcg = 0.0
#         for i in range(len(res)):
#             one_ndcg += res[i] / math.log(i + 2, 2)
#         ndcg += one_ndcg
#     return ndcg


# def hit_k(topk_results, k):
#     hit = 0.0
#     for row in topk_results:
#         res = row[:k]
#         if sum(res) > 0:
#             hit += 1
#     return hit




# import math

# def get_topk_results(predictions, scores, targets, k, all_items=None):
#     """
#     返回:
#       results: List[List[int]]  # 用于hit/ndcg计算
#       top1_pairs: List[dict]    # 每条样本top1预测编码 & 正确答案编码: {"pred":..., "target":...}
#     """
#     results = []
#     top1_pairs = []

#     B = len(targets)
#     predictions = [_.split("Response:")[-1] for _ in predictions]
#     predictions = [_.strip().replace(" ", "") for _ in predictions]

#     if all_items is not None:
#         for i, seq in enumerate(predictions):
#             if seq not in all_items:
#                 scores[i] = -1000

#     for b in range(B):
#         batch_seqs = predictions[b * k: (b + 1) * k]
#         batch_scores = scores[b * k: (b + 1) * k]

#         pairs = [(a, s) for a, s in zip(batch_seqs, batch_scores)]
#         sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

#         target_item = targets[b]

#         # ✅ 只记录“最高分预测编码”和“答案编码”
#         top1_pred = sorted_pairs[0][0]
#         top1_pairs.append({
#             "pred": top1_pred,
#             "target": target_item
#         })

#         one_results = []
#         for pred, _ in sorted_pairs:
#             one_results.append(1 if pred == target_item else 0)

#         results.append(one_results)

#     return results, top1_pairs


# def get_metrics_results(topk_results, metrics):
#     res = {}
#     for m in metrics:
#         if m.lower().startswith("hit"):
#             k = int(m.split("@")[1])
#             res[m] = hit_k(topk_results, k)
#         elif m.lower().startswith("ndcg"):
#             k = int(m.split("@")[1])
#             res[m] = ndcg_k(topk_results, k)
#         else:
#             raise NotImplementedError
#     return res


# def ndcg_k(topk_results, k):
#     ndcg = 0.0
#     for row in topk_results:
#         res = row[:k]
#         one_ndcg = 0.0
#         for i in range(len(res)):
#             one_ndcg += res[i] / math.log(i + 2, 2)
#         ndcg += one_ndcg
#     return ndcg


# def hit_k(topk_results, k):
#     hit = 0.0
#     for row in topk_results:
#         res = row[:k]
#         if sum(res) > 0:
#             hit += 1
#     return hit










# import math
# import re

# # 抓取形如 <a_3> <b_69> ...
# TOKEN_RE = re.compile(r"<[abcd]_\d+>")

# def keep_last4_codes(s: str) -> str:
#     """
#     将预测/目标字符串标准化为“后四位 abcd”。
#     - 如果能解析到 >=5 个 token（aabcd），返回第2~5个拼接
#     - 如果能解析到 4 个 token（abcd），返回这4个拼接
#     - 解析不到则返回原字符串（你也可以改成 '' 让它必然不命中）
#     """
#     s = s.strip().replace(" ", "")
#     tokens = TOKEN_RE.findall(s)
#     if len(tokens) >= 5:
#         return "".join(tokens[1:5])  # drop first a, keep next 4
#     if len(tokens) == 4:
#         return "".join(tokens)
#     return s  # fallback

# def get_topk_results(predictions, scores, targets, k, all_items=None):
#     results = []
#     B = len(targets)

#     # 1) 取 Response 之后的部分 + 基础清洗
#     predictions = [_.split("Response:")[-1] for _ in predictions]
#     predictions = [_.strip().replace(" ", "") for _ in predictions]

#     # 2) 只保留后四位用于评估
#     predictions = [keep_last4_codes(_) for _ in predictions]
#     targets = [keep_last4_codes(str(_)) for _ in targets]

#     # 3) filter_items 也要基于“后四位”的 item 集合
#     if all_items is not None:
#         # 确保 all_items 也是“后四位”标准形式，否则会误过滤
#         all_items_last4 = set(keep_last4_codes(str(x)) for x in all_items)
#         for i, seq in enumerate(predictions):
#             if seq not in all_items_last4:
#                 scores[i] = -1000

#     # 4) 原逻辑不变：按字符串全等比对 target
#     for b in range(B):
#         batch_seqs = predictions[b * k: (b + 1) * k]
#         batch_scores = scores[b * k: (b + 1) * k]

#         pairs = [(a, b_) for a, b_ in zip(batch_seqs, batch_scores)]
#         sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

#         target_item = targets[b]
#         one_results = []
#         for pred_seq, _score in sorted_pairs:
#             one_results.append(1 if pred_seq == target_item else 0)

#         results.append(one_results)

#     return results

# # 下面 metrics 代码不需要改
# def get_metrics_results(topk_results, metrics):
#     res = {}
#     for m in metrics:
#         if m.lower().startswith("hit"):
#             k = int(m.split("@")[1])
#             res[m] = hit_k(topk_results, k)
#         elif m.lower().startswith("ndcg"):
#             k = int(m.split("@")[1])
#             res[m] = ndcg_k(topk_results, k)
#         else:
#             raise NotImplementedError
#     return res

# def ndcg_k(topk_results, k):
#     ndcg = 0.0
#     for row in topk_results:
#         res = row[:k]
#         one_ndcg = 0.0
#         for i in range(len(res)):
#             one_ndcg += res[i] / math.log(i + 2, 2)
#         ndcg += one_ndcg
#     return ndcg

# def hit_k(topk_results, k):
#     hit = 0.0
#     for row in topk_results:
#         res = row[:k]
#         if sum(res) > 0:
#             hit += 1
#     return hit







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
