import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import pdb
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from pycocotools.coco import COCO
from tvd.pycocoevalcap.eval import COCOEvalCap
from .dist_utils import all_gather


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """

    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def contrastive_loss(x, temperature, cl_mask):
    i_logsm = F.log_softmax(x * temperature, dim=1)
    j_logsm = F.log_softmax(x.t() * temperature, dim=1)

    # sum over positives
    # idiag = torch.diag(i_logsm)
    idiag = i_logsm * cl_mask
    loss_i = idiag.sum() / len(idiag)

    #  jdiag = torch.diag(j_logsm)
    jdiag = j_logsm * cl_mask
    loss_j = jdiag.sum() / len(jdiag)
    return -loss_i - loss_j


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, mlm_logits.shape[-1]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack([ti if itm_labels[i] == 1 else fi for i, (ti, fi) in enumerate(zip(bti, bfi))])
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_irtr(pl_module, batch):
    _bs, _t, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)

    subtitle_text_ids = batch["subtitle_text_ids"].unsqueeze(1).repeat(1, false_len + 1, 1)
    subtitle_text_masks = batch["subtitle_text_masks"].unsqueeze(1).repeat(1, false_len + 1, 1)

    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _t, _c, _h, _w)
    text_labels_lm = batch["text_labels_lm"].unsqueeze(1).repeat(1, false_len + 1, 1)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs t c h w -> (bs fs) t c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
            "text_labels_lm": rearrange(text_labels_lm, "bs fs tl -> (bs fs) tl"),
            "subtitle_text_ids": rearrange(subtitle_text_ids, "bs fs tl -> (bs fs) tl"),
            "subtitle_text_masks": rearrange(subtitle_text_masks, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    if pl_module.hparams.config["test_only"]:
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()
    else:
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=16,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    if pl_module.hparams.config["test_only"]:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(image_only=True)
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        for k, v in _b.items():
            if isinstance(v, torch.Tensor):
                _b[k] = v.to(pl_module.device)
        text_preload.append(_b)

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b["image"][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="img rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1, 1).to(device=txt_batch["text_ids"].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        txt_batch,
                        img=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()

    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def evaluate(cache_path, ann_path, dist=True):
    coco = COCO(ann_path)
    valids = coco.getImgIds()

    rank = torch.distributed.get_rank() if dist else 0

    if rank == 0:
        preds = json.load(open(cache_path))
        # filter results to only those in MSCOCO validation set
        preds_filt = [p for p in preds if p["image_id"] in valids]
        print("using %d/%d predictions" % (len(preds_filt), len(preds)))
        json.dump(
            preds_filt, open(cache_path, "w"), ensure_ascii=False
        )  # serialize to temporary json file. Sigh, COCO API...
    if dist:
        torch.distributed.barrier()
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()
    eval_res = cocoEval.eval
    return eval_res


@torch.no_grad()
def compute_mlm_caption(pl_module):
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=32,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    prompt = pl_module.hparams.config.get("caption_prompt", None)
    max_text_len = 100
    stop_word_ids = [image_dset.tokenizer.eos_token_id]
    text_token_start_idx = 0
    if pl_module.config.get("add_new_bos_token", False):
        stop_word_ids.append(image_dset.tokenizer.convert_tokens_to_ids("<eos>"))
        text_token_start_idx = 0
    device = pl_module.device
    rank = torch.distributed.get_rank()

    if prompt is not None:
        prompt_tokens = image_dset.tokenizer.tokenize(prompt)
        fake_start_ids = image_dset.tokenizer.convert_tokens_to_ids(prompt_tokens)

    results = []

    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        B = _b["image"][0].shape[0]
        img_ids = _b["img_index"]
        pred_ids = None
        pred_ids_list = []
        stop_flag = torch.full((B,), 0).bool().to(device)
        ava_len = torch.full((B,), 0).to(device)
        mask_ids = torch.full((B, 1), image_dset.tokenizer.mask_token_id).long().to(device)
        for t in range(max_text_len):
            if t == 0:
                if image_dset.tokenizer.bos_token_id is not None:
                    bos_id = image_dset.tokenizer.bos_token_id
                else:
                    bos_id = image_dset.tokenizer.convert_tokens_to_ids("[CLS]")
                text_ids = torch.full((B, 1), bos_id).long().to(device)
                text_masks = torch.full((B, 1), 1).long().to(device)

                if prompt is not None:
                    pred_ids = torch.tensor(fake_start_ids)[None].long().to(device).repeat(B, 1)
                    pred_ids_list.extend([pred_ids[:, i] for i in range(pred_ids.shape[1])])
                    text_masks = torch.cat([text_masks, torch.full_like(pred_ids, 1)], dim=1).long().to(device)
                    text_ids = torch.cat((text_ids, pred_ids), dim=-1)
            else:
                text_ids = torch.cat((text_ids[:, :-1], pred_ids[:, None]), dim=-1)

            text_ids = torch.cat((text_ids, mask_ids), dim=-1)
            text_masks = torch.cat([text_masks, 1 - stop_flag[:, None].long()], dim=-1)
            _b["image"] = [__b.to(device) for __b in _b["image"]]
            _b["text_ids"] = text_ids
            _b["text_masks"] = text_masks
            _b["text_labels"] = None
            _b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in _b.items()}

            infer = pl_module.infer(_b)
            mlm_logits = pl_module.mlm_score(infer["text_feats"])[:, -1]
            pred_ids = mlm_logits.argmax(1)
            for stop_word_id in stop_word_ids:
                stop_flag = stop_flag | (pred_ids == stop_word_id)
            ava_len = ava_len + (1 - stop_flag.int())
            pred_ids_list.append(pred_ids)

            if (1 - stop_flag.int()).sum() == 0:
                break
        pred_ids_list = torch.stack(pred_ids_list, dim=-1).cpu().numpy().tolist()

        pred_texts = [
            image_dset.tokenizer.decode(pred_id[text_token_start_idx : ava_len[i] + text_token_start_idx])
            for i, pred_id in enumerate(pred_ids_list)
        ]
        for idx, text in enumerate(pred_texts):
            image_id = image_dset.get_sample_id(img_ids[idx])
            results.extend([{"image_id": image_id, "caption": text.strip().strip(".")}])

    rets = results
    exp_path = pl_module.config["exp_path"]
    result_path_rank = os.path.join(exp_path, f"caption_{rank}.json")
    with open(result_path_rank, "w") as fp:
        print("!!! saving caption results to {}".format(result_path_rank))
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()
    result_path = os.path.join(exp_path, f"caption.json")
    if rank == 0:
        jsons = list()
        paths = list(glob.glob(os.path.join(exp_path, "caption_*.json")))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        image_ids = []
        new_jsons = []
        for img_info in jsons:
            if img_info["image_id"] in image_ids:
                continue
            new_jsons.append(img_info)
            image_ids.append(img_info["image_id"])

        os.makedirs("result", exist_ok=True)
        with open(result_path, "w") as fp:
            print("!!! saving final caption results to {}".format(result_path))
            json.dump(new_jsons, fp, indent=4, ensure_ascii=False)

    torch.distributed.barrier()
    os.remove(os.path.join(exp_path, f"caption_{rank}.json"))
    print("!!! deleting caption results at {}".format(result_path_rank))

    scores = evaluate(result_path, pl_module.hparams.config["gt_caption_anno_path"])

    print(scores)
    torch.distributed.barrier()
    b4, m, c, s = (
        scores["Bleu_4"],
        scores["METEOR"],
        scores["CIDEr"],
        scores.get("SPICE", 0),
    )
    return b4, m, c, s


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def arc_test_step(pl_module, batch, output):
    return output


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
