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
