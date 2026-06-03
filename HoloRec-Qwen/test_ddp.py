# # import argparse
# # import json
# # import os
# # import sys

# # import torch
# # import transformers
# # import torch.distributed as dist
# # from torch.utils.data.distributed import DistributedSampler
# # from torch.nn.parallel import DistributedDataParallel
# # from peft import PeftModel
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm
# # from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

# # from utils import *
# # from collator import TestCollator
# # from prompt import all_prompt
# # from evaluate import get_topk_results, get_metrics_results
# # # import os

# # # os.environ['MASTER_ADDR'] = 'localhost'
# # # os.environ['MASTER_PORT'] = '5678'


# # def test_ddp(args):

# #     set_seed(args.seed)
# #     world_size = int(os.environ.get("WORLD_SIZE", 1))
# #     local_rank = int(os.environ.get("LOCAL_RANK") or 0)
# #     torch.cuda.set_device(local_rank)
# #     if local_rank == 0:
# #         print(vars(args))

# #     dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

# #     device_map = {"": local_rank}
# #     device = torch.device("cuda",local_rank)

# #     tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
# #     if args.lora:
# #         model = LlamaForCausalLM.from_pretrained(
# #             args.base_model,
# #             torch_dtype=torch.bfloat16,
# #             low_cpu_mem_usage=True,
# #             device_map=device_map,
# #         )
# #         model.resize_token_embeddings(len(tokenizer))
# #         model = PeftModel.from_pretrained(
# #             model,
# #             args.ckpt_path,
# #             torch_dtype=torch.bfloat16,
# #             device_map=device_map,
# #         )
# #     else:
# #         model = LlamaForCausalLM.from_pretrained(
# #             args.ckpt_path,
# #             torch_dtype=torch.bfloat16,              
# #             load_in_8bit=True,
# #             low_cpu_mem_usage=True,
# #             device_map=device_map,
# #         )
# #     # assert model.config.vocab_size == len(tokenizer)
# #     model = DistributedDataParallel(model, device_ids=[local_rank])

# #     if args.test_prompt_ids == "all":
# #         if args.test_task.lower() == "seqrec":
# #             prompt_ids = range(len(all_prompt["seqrec"]))
# #         elif args.test_task.lower() == "itemsearch":
# #             prompt_ids = range(len(all_prompt["itemsearch"]))
# #         elif args.test_task.lower() == "fusionseqrec":
# #             prompt_ids = range(len(all_prompt["fusionseqrec"]))
# #     else:
# #         prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

# #     test_data = load_test_dataset(args)
# #     ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)

# #     test_data = load_test_dataset(args)
# #     collator = TestCollator(args, tokenizer)
# #     all_items = test_data.get_all_items()


# #     prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)


# #     test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
# #                              sampler=ddp_sampler, num_workers=2, pin_memory=True)

# #     if local_rank == 0:
# #         print("data num:", len(test_data))

# #     model.eval()

# #     metrics = args.metrics.split(",")
# #     all_prompt_results = []
# #     with torch.no_grad():

# #         for prompt_id in prompt_ids:

# #             if local_rank == 0:
# #                 print("Start prompt: ",prompt_id)

# #             test_loader.dataset.set_prompt(prompt_id)
# #             metrics_results = {}
# #             total = 0

# #             for step, batch in enumerate(tqdm(test_loader)):
# #                 inputs = batch[0].to(device)
# #                 targets = batch[1]
# #                 bs = len(targets)
# #                 num_beams = args.num_beams
# #                 while True:
# #                     try:
# #                         output = model.module.generate(
# #                             input_ids=inputs["input_ids"],
# #                             attention_mask=inputs["attention_mask"],
# #                             max_new_tokens=10,
# #                             prefix_allowed_tokens_fn=prefix_allowed_tokens,
# #                             num_beams=num_beams,
# #                             num_return_sequences=num_beams,
# #                             output_scores=True,
# #                             return_dict_in_generate=True,
# #                             early_stopping=True,
# #                         )
# #                         break
# #                     except torch.cuda.OutOfMemoryError as e:
# #                         print("Out of memory!")
# #                         num_beams = num_beams -1
# #                         print("Beam:", num_beams)
# #                     except Exception:
# #                         raise RuntimeError

# #                 output_ids = output["sequences"]
# #                 scores = output["sequences_scores"]

# #                 output = tokenizer.batch_decode(
# #                     output_ids, skip_special_tokens=True
# #                 )

# #                 topk_res = get_topk_results(output, scores, targets, num_beams,
# #                                             all_items=all_items if args.filter_items else None)

# #                 bs_gather_list = [None for _ in range(world_size)]
# #                 dist.all_gather_object(obj=bs, object_list=bs_gather_list)
# #                 total += sum(bs_gather_list)
# #                 res_gather_list = [None for _ in range(world_size)]
# #                 dist.all_gather_object(obj=topk_res, object_list=res_gather_list)


# #                 if local_rank == 0:
# #                     all_device_topk_res = []
# #                     for ga_res in res_gather_list:
# #                         all_device_topk_res += ga_res
# #                     batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
# #                     for m, res in batch_metrics_res.items():
# #                         if m not in metrics_results:
# #                             metrics_results[m] = res
# #                         else:
# #                             metrics_results[m] += res

# #                     if (step + 1) % 50 == 0:
# #                         temp = {}
# #                         for m in metrics_results:
# #                             temp[m] = metrics_results[m] / total
# #                         print(temp)

# #                 dist.barrier()

# #             if local_rank == 0:
# #                 for m in metrics_results:
# #                     metrics_results[m] = metrics_results[m] / total

# #                 all_prompt_results.append(metrics_results)
# #                 print("======================================================")
# #                 print("Prompt {} results: ".format(prompt_id), metrics_results)
# #                 print("======================================================")
# #                 print("")

# #             dist.barrier()

# #     dist.barrier()

# #     if local_rank == 0:
# #         mean_results = {}
# #         min_results = {}
# #         max_results = {}

# #         for m in metrics:
# #             all_res = [_[m] for _ in all_prompt_results]
# #             mean_results[m] = sum(all_res)/len(all_res)
# #             min_results[m] = min(all_res)
# #             max_results[m] = max(all_res)

# #         print("======================================================")
# #         print("Mean results: ", mean_results)
# #         print("Min results: ", min_results)
# #         print("Max results: ", max_results)
# #         print("======================================================")


# #         save_data={}
# #         save_data["test_prompt_ids"] = args.test_prompt_ids
# #         save_data["mean_results"] = mean_results
# #         save_data["min_results"] = min_results
# #         save_data["max_results"] = max_results
# #         save_data["all_prompt_results"] = all_prompt_results

# #         with open(args.results_file, "w") as f:
# #             json.dump(save_data, f, indent=4)
# #         print("Save file: ", args.results_file)



# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="LLMRec_test")
# #     parser = parse_global_args(parser)
# #     parser = parse_dataset_args(parser)
# #     parser = parse_test_args(parser)

# #     args = parser.parse_args()
# #     test_ddp(args)





# # import argparse
# # import json
# # import os
# # import sys

# # import torch
# # import transformers
# # import torch.distributed as dist
# # from torch.utils.data.distributed import DistributedSampler
# # from torch.nn.parallel import DistributedDataParallel
# # from peft import PeftModel
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm
# # from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

# # from utils import *
# # from collator import TestCollator
# # from prompt import all_prompt
# # from evaluate import get_topk_results, get_metrics_results  # get_topk_results returns (topk_res, top1_pairs)

# # # ✅ 固定保存路径（只在rank0写）
# # FIXED_SAVE_PATH = "/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/res.json"


# # def test_ddp(args):
# #     set_seed(args.seed)
# #     world_size = int(os.environ.get("WORLD_SIZE", 1))
# #     local_rank = int(os.environ.get("LOCAL_RANK") or 0)
# #     torch.cuda.set_device(local_rank)
# #     if local_rank == 0:
# #         print(vars(args))

# #     dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

# #     device_map = {"": local_rank}
# #     device = torch.device("cuda", local_rank)

# #     tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
# #     if args.lora:
# #         model = LlamaForCausalLM.from_pretrained(
# #             args.base_model,
# #             torch_dtype=torch.bfloat16,
# #             low_cpu_mem_usage=True,
# #             device_map=device_map,
# #         )
# #         model.resize_token_embeddings(len(tokenizer))
# #         model = PeftModel.from_pretrained(
# #             model,
# #             args.ckpt_path,
# #             torch_dtype=torch.bfloat16,
# #             device_map=device_map,
# #         )
# #     else:
# #         model = LlamaForCausalLM.from_pretrained(
# #             args.ckpt_path,
# #             torch_dtype=torch.bfloat16,
# #             load_in_8bit=True,
# #             low_cpu_mem_usage=True,
# #             device_map=device_map,
# #         )

# #     model = DistributedDataParallel(model, device_ids=[local_rank])

# #     if args.test_prompt_ids == "all":
# #         if args.test_task.lower() == "seqrec":
# #             prompt_ids = range(len(all_prompt["seqrec"]))
# #         elif args.test_task.lower() == "itemsearch":
# #             prompt_ids = range(len(all_prompt["itemsearch"]))
# #         elif args.test_task.lower() == "fusionseqrec":
# #             prompt_ids = range(len(all_prompt["fusionseqrec"]))
# #         else:
# #             raise ValueError(f"Unknown test_task: {args.test_task}")
# #     else:
# #         prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

# #     test_data = load_test_dataset(args)
# #     ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)

# #     test_data = load_test_dataset(args)
# #     collator = TestCollator(args, tokenizer)
# #     all_items = test_data.get_all_items()

# #     prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

# #     test_loader = DataLoader(
# #         test_data,
# #         batch_size=args.test_batch_size,
# #         collate_fn=collator,
# #         sampler=ddp_sampler,
# #         num_workers=2,
# #         pin_memory=True,
# #     )

# #     if local_rank == 0:
# #         print("data num:", len(test_data))

# #     model.eval()

# #     metrics = args.metrics.split(",")
# #     all_prompt_results = []

# #     # ✅ 最终要写入的 (pred,target) 列表，只在rank0收集
# #     all_best_pairs_rank0 = []

# #     with torch.no_grad():
# #         for prompt_id in prompt_ids:
# #             if local_rank == 0:
# #                 print("Start prompt: ", prompt_id)

# #             test_loader.dataset.set_prompt(prompt_id)
# #             metrics_results = {}
# #             total = 0

# #             for step, batch in enumerate(tqdm(test_loader)):
# #                 inputs = batch[0].to(device)
# #                 targets = batch[1]
# #                 bs = len(targets)

# #                 num_beams = args.num_beams
# #                 while True:
# #                     try:
# #                         output = model.module.generate(
# #                             input_ids=inputs["input_ids"],
# #                             attention_mask=inputs["attention_mask"],
# #                             max_new_tokens=10,
# #                             prefix_allowed_tokens_fn=prefix_allowed_tokens,
# #                             num_beams=num_beams,
# #                             num_return_sequences=num_beams,
# #                             output_scores=True,
# #                             return_dict_in_generate=True,
# #                             early_stopping=True,
# #                         )
# #                         break
# #                     except torch.cuda.OutOfMemoryError:
# #                         print("Out of memory!")
# #                         num_beams = num_beams - 1
# #                         print("Beam:", num_beams)
# #                     except Exception:
# #                         raise RuntimeError

# #                 output_ids = output["sequences"]
# #                 scores = output["sequences_scores"]

# #                 decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

# #                 # ✅ 同时获得topk_res和top1(pred,target)
# #                 topk_res, top1_pairs = get_topk_results(
# #                     decoded, scores, targets, num_beams,
# #                     all_items=all_items if args.filter_items else None
# #                 )

# #                 # 原逻辑：统计 total
# #                 bs_gather_list = [None for _ in range(world_size)]
# #                 dist.all_gather_object(obj=bs, object_list=bs_gather_list)
# #                 total += sum(bs_gather_list)

# #                 # 原逻辑：gather topk_res
# #                 res_gather_list = [None for _ in range(world_size)]
# #                 dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

# #                 # ✅ 新增：gather top1_pairs
# #                 pair_gather_list = [None for _ in range(world_size)]
# #                 dist.all_gather_object(obj=top1_pairs, object_list=pair_gather_list)

# #                 if local_rank == 0:
# #                     # 汇总所有设备topk来算指标
# #                     all_device_topk_res = []
# #                     for ga_res in res_gather_list:
# #                         all_device_topk_res += ga_res

# #                     batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
# #                     for m, res in batch_metrics_res.items():
# #                         if m not in metrics_results:
# #                             metrics_results[m] = res
# #                         else:
# #                             metrics_results[m] += res

# #                     # ✅ 汇总所有设备 top1(pred,target)
# #                     for part in pair_gather_list:
# #                         all_best_pairs_rank0.extend(part)

# #                     if (step + 1) % 50 == 0:
# #                         temp = {}
# #                         for m in metrics_results:
# #                             temp[m] = metrics_results[m] / total
# #                         print(temp)

# #                 dist.barrier()

# #             if local_rank == 0:
# #                 for m in metrics_results:
# #                     metrics_results[m] = metrics_results[m] / total

# #                 all_prompt_results.append(metrics_results)
# #                 print("======================================================")
# #                 print("Prompt {} results: ".format(prompt_id), metrics_results)
# #                 print("======================================================")
# #                 print("")

# #             dist.barrier()

# #     dist.barrier()

# #     if local_rank == 0:
# #         # 写固定路径
# #         os.makedirs(os.path.dirname(FIXED_SAVE_PATH), exist_ok=True)
# #         with open(FIXED_SAVE_PATH, "w", encoding="utf-8") as f:
# #             json.dump(
# #                 {
# #                     "ckpt_path": args.ckpt_path,
# #                     "test_prompt_ids": args.test_prompt_ids,
# #                     "metrics": args.metrics,
# #                     "num_pairs": len(all_best_pairs_rank0),
# #                     "pairs": all_best_pairs_rank0,   # 只包含 pred 和 target
# #                 },
# #                 f,
# #                 ensure_ascii=False,
# #                 indent=2,
# #             )
# #         print("Save best(pred,target) pairs to: ", FIXED_SAVE_PATH)

# #         # 你原来的汇总指标输出/保存保持不变
# #         mean_results = {}
# #         min_results = {}
# #         max_results = {}

# #         for m in metrics:
# #             all_res = [_[m] for _ in all_prompt_results]
# #             mean_results[m] = sum(all_res) / len(all_res)
# #             min_results[m] = min(all_res)
# #             max_results[m] = max(all_res)

# #         print("======================================================")
# #         print("Mean results: ", mean_results)
# #         print("Min results: ", min_results)
# #         print("Max results: ", max_results)
# #         print("======================================================")

# #         save_data = {}
# #         save_data["test_prompt_ids"] = args.test_prompt_ids
# #         save_data["mean_results"] = mean_results
# #         save_data["min_results"] = min_results
# #         save_data["max_results"] = max_results
# #         save_data["all_prompt_results"] = all_prompt_results

# #         with open(args.results_file, "w", encoding="utf-8") as f:
# #             json.dump(save_data, f, indent=4, ensure_ascii=False)
# #         print("Save file: ", args.results_file)


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="LLMRec_test")
# #     parser = parse_global_args(parser)
# #     parser = parse_dataset_args(parser)
# #     parser = parse_test_args(parser)

# #     args = parser.parse_args()
# #     test_ddp(args)

# import argparse
# import json
# import os

# import torch
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel

# from utils import *
# from collator import TestCollator
# from prompt import all_prompt
# from evaluate import get_topk_results, get_metrics_results

# # ✅ 固定保存路径（只在rank0写）
# FIXED_SAVE_PATH = "/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/res.json"


# def build_model_and_tokenizer(args, device_map):
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.ckpt_path,
#         trust_remote_code=True,
#         use_fast=True,
#     )

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

#     dtype = torch.bfloat16

#     if args.lora:
#         base = AutoModelForCausalLM.from_pretrained(
#             args.base_model,
#             device_map=device_map,
#             dtype=dtype,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#         )
#         base.resize_token_embeddings(len(tokenizer))

#         model = PeftModel.from_pretrained(
#             base,
#             args.ckpt_path,
#             device_map=device_map,
#             dtype=dtype,
#         )
#     else:
#         quant_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#             llm_int8_threshold=6.0,
#             llm_int8_has_fp16_weight=False,
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             args.ckpt_path,
#             device_map=device_map,
#             quantization_config=quant_config,
#             dtype=dtype,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#         )

#     return tokenizer, model


# def test_ddp(args):
#     set_seed(args.seed)

#     world_size = int(os.environ.get("WORLD_SIZE", 1))
#     local_rank = int(os.environ.get("LOCAL_RANK") or 0)

#     torch.cuda.set_device(local_rank)
#     if local_rank == 0:
#         print(vars(args))

#     dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

#     device_map = {"": local_rank}
#     device = torch.device("cuda", local_rank)

#     tokenizer, model = build_model_and_tokenizer(args, device_map=device_map)

#     model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
#     model.eval()

#     # prompt ids
#     if args.test_prompt_ids == "all":
#         if args.test_task.lower() == "seqrec":
#             prompt_ids = range(len(all_prompt["seqrec"]))
#         elif args.test_task.lower() == "itemsearch":
#             prompt_ids = range(len(all_prompt["itemsearch"]))
#         elif args.test_task.lower() == "fusionseqrec":
#             prompt_ids = range(len(all_prompt["fusionseqrec"]))
#         else:
#             raise ValueError(f"Unknown test_task: {args.test_task}")
#     else:
#         prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

#     test_data = load_test_dataset(args)
#     ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)

#     collator = TestCollator(args, tokenizer)
#     all_items = test_data.get_all_items()

#     # ✅ dataset 返回的基础 prefix_fn（我们会在每个 batch 包一层，把 prompt_len 固定住）
#     base_prefix_fn = test_data.get_prefix_allowed_tokens_fn(tokenizer)

#     test_loader = DataLoader(
#         test_data,
#         batch_size=args.test_batch_size,
#         collate_fn=collator,
#         sampler=ddp_sampler,
#         num_workers=2,
#         pin_memory=True,
#     )

#     if local_rank == 0:
#         print("data num:", len(test_data))

#     metrics = args.metrics.split(",")
#     all_prompt_results = []
#     all_best_pairs_rank0 = []

#     with torch.no_grad():
#         for prompt_id in prompt_ids:
#             if local_rank == 0:
#                 print("Start prompt: ", prompt_id)

#             test_loader.dataset.set_prompt(prompt_id)
#             metrics_results = {}
#             total = 0

#             for step, batch in enumerate(tqdm(test_loader)):
#                 inputs = batch[0].to(device)
#                 targets = batch[1]
#                 bs = len(targets)

#                 # ✅ 当前 batch 的 prompt_len（对所有 beam 都是常量）
#                 prompt_len = inputs["input_ids"].shape[1]

#                 # ✅ 关键：用闭包把 prompt_len 传给 prefix_fn，避免 beam/batch_id 错位
#                 def prefix_fn(batch_id, sentence):
#                     try:
#                         # 新版（你按我建议改过 data.py 的签名）
#                         return base_prefix_fn(batch_id, sentence, prompt_len=prompt_len)
#                     except TypeError:
#                         # 兼容旧版：如果 base_prefix_fn 只有两个参数
#                         # 这种情况下仍然可能有 beam 错位问题，但至少不崩
#                         return base_prefix_fn(batch_id, sentence)

#                 num_beams = args.num_beams
#                 while True:
#                     try:
#                         output = model.module.generate(
#                             input_ids=inputs["input_ids"],
#                             attention_mask=inputs["attention_mask"],
#                             max_new_tokens=10,
#                             prefix_allowed_tokens_fn=prefix_fn,
#                             num_beams=num_beams,
#                             num_return_sequences=num_beams,
#                             output_scores=True,
#                             return_dict_in_generate=True,
#                             early_stopping=True,
#                         )
#                         break
#                     except torch.cuda.OutOfMemoryError:
#                         print("Out of memory!")
#                         num_beams -= 1
#                         print("Beam:", num_beams)
#                         if num_beams <= 0:
#                             raise RuntimeError("num_beams reduced to 0 due to OOM.")
#                     except Exception as e:
#                         raise RuntimeError(e)

#                 output_ids = output["sequences"]
#                 scores = output["sequences_scores"]

#                 # ✅ 只 decode 新生成部分
#                 gen_ids = output_ids[:, prompt_len:]
#                 decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

#                 # ✅ debug：看看 batch 里空的比例（只 rank0，前几步打印）
#                 if local_rank == 0 and step < 3:
#                     empty_cnt = sum(1 for x in decoded if x.strip() == "")
#                     print(f"[debug] empty decoded: {empty_cnt}/{len(decoded)} ; sample0={repr(decoded[0])}")

#                 # topk + top1 pairs
#                 topk_res, top1_pairs = get_topk_results(
#                     decoded, scores, targets, num_beams,
#                     all_items=all_items if args.filter_items else None
#                 )

#                 # gather bs
#                 bs_gather_list = [None for _ in range(world_size)]
#                 dist.all_gather_object(obj=bs, object_list=bs_gather_list)
#                 total += sum(bs_gather_list)

#                 # gather topk_res
#                 res_gather_list = [None for _ in range(world_size)]
#                 dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

#                 # gather top1_pairs
#                 pair_gather_list = [None for _ in range(world_size)]
#                 dist.all_gather_object(obj=top1_pairs, object_list=pair_gather_list)

#                 if local_rank == 0:
#                     all_device_topk_res = []
#                     for ga_res in res_gather_list:
#                         all_device_topk_res += ga_res

#                     batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
#                     for m, res in batch_metrics_res.items():
#                         metrics_results[m] = metrics_results.get(m, 0) + res

#                     for part in pair_gather_list:
#                         all_best_pairs_rank0.extend(part)

#                     if (step + 1) % 50 == 0:
#                         temp = {m: metrics_results[m] / total for m in metrics_results}
#                         print(temp)

#                 dist.barrier()

#             if local_rank == 0:
#                 for m in metrics_results:
#                     metrics_results[m] = metrics_results[m] / total

#                 all_prompt_results.append(metrics_results)
#                 print("======================================================")
#                 print("Prompt {} results: ".format(prompt_id), metrics_results)
#                 print("======================================================")
#                 print("")

#             dist.barrier()

#     dist.barrier()

#     if local_rank == 0:
#         os.makedirs(os.path.dirname(FIXED_SAVE_PATH), exist_ok=True)
#         with open(FIXED_SAVE_PATH, "w", encoding="utf-8") as f:
#             json.dump(
#                 {
#                     "ckpt_path": args.ckpt_path,
#                     "test_prompt_ids": args.test_prompt_ids,
#                     "metrics": args.metrics,
#                     "num_pairs": len(all_best_pairs_rank0),
#                     "pairs": all_best_pairs_rank0,
#                 },
#                 f,
#                 ensure_ascii=False,
#                 indent=2,
#             )
#         print("Save best(pred,target) pairs to: ", FIXED_SAVE_PATH)

#         mean_results = {}
#         min_results = {}
#         max_results = {}

#         for m in metrics:
#             all_res = [_[m] for _ in all_prompt_results]
#             mean_results[m] = sum(all_res) / len(all_res)
#             min_results[m] = min(all_res)
#             max_results[m] = max(all_res)

#         print("======================================================")
#         print("Mean results: ", mean_results)
#         print("Min results: ", min_results)
#         print("Max results: ", max_results)
#         print("======================================================")

#         save_data = {
#             "test_prompt_ids": args.test_prompt_ids,
#             "mean_results": mean_results,
#             "min_results": min_results,
#             "max_results": max_results,
#             "all_prompt_results": all_prompt_results,
#         }

#         with open(args.results_file, "w", encoding="utf-8") as f:
#             json.dump(save_data, f, indent=4, ensure_ascii=False)
#         print("Save file: ", args.results_file)

#     dist.barrier()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="LLMRec_test")
#     parser = parse_global_args(parser)
#     parser = parse_dataset_args(parser)
#     parser = parse_test_args(parser)

#     args = parser.parse_args()
#     test_ddp(args)





import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results


def build_model_and_tokenizer(args, device_map):
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    dtype = torch.bfloat16

    if args.lora:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=device_map,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(
            base,
            args.ckpt_path,
            device_map=device_map,
            dtype=dtype,
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            device_map=device_map,
            quantization_config=quant_config,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    return tokenizer, model


def gather_object_list(local_obj, world_size):
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_obj)
    merged = []
    for part in gathered:
        if part is None:
            continue
        if isinstance(part, list):
            merged.extend(part)
        else:
            merged.append(part)
    return merged


def test_ddp(args):
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    tokenizer, model = build_model_and_tokenizer(args, device_map=device_map)

    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    model.eval()

    # prompt ids
    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
        else:
            raise ValueError(f"Unknown test_task: {args.test_task}")
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(
        test_data,
        num_replicas=world_size,
        rank=local_rank,
        drop_last=True,
        shuffle=False,
    )

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    # 这里是 fine/coarse 并集约束
    base_prefix_fn = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        sampler=ddp_sampler,
        num_workers=2,
        pin_memory=True,
    )

    if local_rank == 0:
        print("data num:", len(test_data))

    metrics = args.metrics.split(",")
    all_prompt_results = []
    all_pairs_rank0 = []
    all_txt_lines_rank0 = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            if local_rank == 0:
                print("Start prompt: ", prompt_id)

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            local_pairs = []
            local_txt_lines = []

            for step, batch in enumerate(tqdm(test_loader, disable=(local_rank != 0))):
                inputs = batch[0].to(device)
                fine_targets = batch[1]
                coarse_targets = batch[2]
                bs = len(fine_targets)

                prompt_len = inputs["input_ids"].shape[1]

                def prefix_fn(batch_id, sentence):
                    try:
                        return base_prefix_fn(batch_id, sentence, prompt_len=prompt_len)
                    except TypeError:
                        return base_prefix_fn(batch_id, sentence)

                num_beams = args.num_beams
                while True:
                    try:
                        output = model.module.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            prefix_allowed_tokens_fn=prefix_fn,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        print("Out of memory!")
                        num_beams -= 1
                        print("Beam:", num_beams)
                        if num_beams <= 0:
                            raise RuntimeError("num_beams reduced to 0 due to OOM.")
                    except Exception as e:
                        raise RuntimeError(e)

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                # 只 decode 新生成部分
                gen_ids = output_ids[:, prompt_len:]
                decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                if local_rank == 0 and step < 3:
                    empty_cnt = sum(1 for x in decoded if x.strip() == "")
                    print(f"[debug] empty decoded: {empty_cnt}/{len(decoded)} ; sample0={repr(decoded[0])}")

                topk_res, top1_pairs = get_topk_results(
                    decoded,
                    scores,
                    fine_targets,
                    coarse_targets,
                    num_beams,
                    all_items=all_items if args.filter_items else None,
                )

                # 本 rank 保存 txt 行
                if args.save_simple_results:
                    for pair in top1_pairs:
                        fine_t = pair["fine_target"]
                        coarse_t = pair["coarse_target"]
                        pred = pair["pred"]

                        local_txt_lines.append(f"{fine_t} | {coarse_t}")
                        local_txt_lines.append(f"{pred}")

                local_pairs.extend(top1_pairs)

                # gather batch size
                bs_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                total += sum(bs_gather_list)

                # gather topk results
                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

                if local_rank == 0:
                    all_device_topk_res = []
                    for ga_res in res_gather_list:
                        all_device_topk_res += ga_res

                    batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                    for m, res in batch_metrics_res.items():
                        metrics_results[m] = metrics_results.get(m, 0) + res

                    if (step + 1) % 50 == 0:
                        temp = {m: metrics_results[m] / total for m in metrics_results}
                        print(temp)

                dist.barrier()

            # gather top1 pairs and txt lines
            gathered_pairs = gather_object_list(local_pairs, world_size)
            gathered_txt_lines = gather_object_list(local_txt_lines, world_size)

            if local_rank == 0:
                all_pairs_rank0.extend(gathered_pairs)
                if args.save_simple_results:
                    all_txt_lines_rank0.extend(gathered_txt_lines)

                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total

                all_prompt_results.append(metrics_results)
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")
                print("")

            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)

        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")

        save_data = {
            "test_prompt_ids": args.test_prompt_ids,
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "all_prompt_results": all_prompt_results,
        }

        results_dir = os.path.dirname(args.results_file)
        if results_dir != "":
            os.makedirs(results_dir, exist_ok=True)

        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        print("Save file: ", args.results_file)

        # 额外保存 json top1 pair
        if args.save_pairs_json:
            pairs_dir = os.path.dirname(args.pairs_json_file)
            if pairs_dir != "":
                os.makedirs(pairs_dir, exist_ok=True)

            with open(args.pairs_json_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ckpt_path": args.ckpt_path,
                        "test_prompt_ids": args.test_prompt_ids,
                        "metrics": args.metrics,
                        "num_pairs": len(all_pairs_rank0),
                        "pairs": all_pairs_rank0,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print("Save pairs json to:", args.pairs_json_file)

        # 保存简洁 txt：两行一个样本
        if args.save_simple_results:
            txt_dir = os.path.dirname(args.simple_results_file)
            if txt_dir != "":
                os.makedirs(txt_dir, exist_ok=True)

            with open(args.simple_results_file, "w", encoding="utf-8") as f:
                for line in all_txt_lines_rank0:
                    f.write(line + "\n")

            print("Save simple txt to:", args.simple_results_file)

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    parser.add_argument("--coarse_index_file", type=str, default=".index.4cu32xi.json")

    parser.add_argument("--save_pairs_json", action="store_true")
    parser.add_argument("--pairs_json_file", type=str, default="/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/results")

    parser.add_argument("--save_simple_results", action="store_true")
    parser.add_argument("--simple_results_file", type=str, default="/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/results/simple_results.txt")

    args = parser.parse_args()
    test_ddp(args)
