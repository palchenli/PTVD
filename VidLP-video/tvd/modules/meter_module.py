import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertModel,
)
from .bert_model import BertCrossLayer
from . import swin_transformer as swin
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel
from .video_transformer import SpaceTimeTransformer
import torch.distributed as dist
from tvd.utils.utils import adapt_vocab_size


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args["world_size"])]
        dist.all_gather(output, tensor)
        ctx.rank = args["rank"]
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )


class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.hparams.config = config

        self.is_clip = (not "swin" in config["vit"]) and (not "SpaceTimeTransformer" in config["vit"])
        self.is_swin = "swin" in config["vit"]
        self.is_videoformer = "SpaceTimeTransformer" in config["vit"]

        if "roberta" in config["tokenizer"]:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
                is_decoder=config["is_causal_mask"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
                is_decoder=config["is_causal_mask"],
            )

        resolution_after = config["image_size"]

        self.all_gather = AllGather_multi.apply
        self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_text_transform.apply(objectives.init_weights)

        self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(3, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config["vit"], resolution_after=resolution_after)
                elif self.is_swin:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True,
                        config=self.hparams.config,
                    )

                if "roberta" in config["tokenizer"]:
                    RobertaModel.from_pretrained(config["tokenizer"], cache_dir=".cache")
                else:
                    BertModel.from_pretrained(config["tokenizer"], cache_dir=".cache")

            torch.distributed.barrier()

        if self.is_videoformer:
            video_params = {}
            num_frames = config["num_frames"]
            pretrained = video_params.get("pretrained", True)
            time_init = video_params.get("time_init", "zeros")
            attention_style = video_params.get("attention_style", "frozen-in-time")
            arch_config = video_params.get("arch_config", "base_patch16_224")
            if arch_config == "base_patch16_224":
                import timm

                vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                model = SpaceTimeTransformer(
                    num_frames=num_frames,
                    time_init=time_init,
                    attention_style=attention_style,
                )
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            # if load_checkpoint in ["", None]:
            vit_checkpoint = vit_model.state_dict()
            model.load_state_dict(vit_checkpoint, strict=False)
            model.fc = nn.Identity()
            self.vit_model = model
        else:
            if self.is_clip:
                self.vit_model = build_model(config["vit"], resolution_after=resolution_after)
            elif self.is_swin:
                self.vit_model = getattr(swin, self.hparams.config["vit"])(
                    pretrained=True,
                    config=self.hparams.config,
                )
                self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.causal_mask = config["is_causal_mask"]

        if config["text_encoder_from_scratch"]:
            te_config = RobertaConfig.from_pretrained(config["tokenizer"])
            if self.causal_mask:
                te_config.is_decoder = True
            self.text_transformer = BertModel(config=te_config)

        elif "roberta" in config["tokenizer"]:
            if self.causal_mask:
                te_config = RobertaConfig.from_pretrained(config["tokenizer"])
                te_config.is_decoder = True
                self.text_transformer = RobertaModel.from_pretrained(
                    config["tokenizer"], config=te_config, cache_dir=".cache"
                )
            else:
                self.text_transformer = RobertaModel.from_pretrained(config["tokenizer"], cache_dir=".cache")
        elif "gpt2" in config["tokenizer"]:
            from transformers import GPT2Tokenizer, GPT2Model

            self.text_transformer = GPT2Model.from_pretrained(config["tokenizer"], cache_dir=".cache")
        else:
            if self.causal_mask:
                te_config = BertConfig.from_pretrained(config["tokenizer"])
                te_config.is_decoder = True
                self.text_transformer = BertModel.from_pretrained(
                    config["tokenizer"], config=te_config, cache_dir=".cache"
                )
            else:
                self.text_transformer = BertModel.from_pretrained(config["tokenizer"], cache_dir=".cache")

        vocab_size = config["vocab_size"]
        if config["add_new_bos_token"]:
            print("add two additional special tokens")
            vocab_size = config["vocab_size"] + 2
            self.text_transformer.resize_token_embeddings(vocab_size)
            bert_config.vocab_size = vocab_size

        if not config["disable_cross_modal_image_layer"]:
            self.cross_modal_image_layers = nn.ModuleList(
                [BertCrossLayer(bert_config) for _ in range(config["num_top_layer"])]
            )
            self.cross_modal_image_layers.apply(objectives.init_weights)

        self.cross_modal_text_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config["num_top_layer"])]
        )
        self.cross_modal_text_layers.apply(objectives.init_weights)

        if not config["disable_cross_modal_image_layer"]:
            self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        # ===================== Downstream ===================== #
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(
                    state_dict,
                    after=resolution_after,
                    patch_size=self.hparams.config["patch_size"],
                )
            elif self.is_swin:
                state_dict = swin_adapt_position_encoding(
                    state_dict,
                    after=resolution_after,
                    before=config["resolution_before"],
                )

            state_dict = adapt_vocab_size(state_dict, vocab_size)
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                self.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(2 * hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(
                    state_dict,
                    after=resolution_after,
                    patch_size=self.hparams.config["patch_size"],
                )
            elif self.is_swin:
                state_dict = swin_adapt_position_encoding(
                    state_dict,
                    after=resolution_after,
                    before=config["resolution_before"],
                )
            state_dict = adapt_vocab_size(state_dict, vocab_size)
            self.load_state_dict(state_dict, strict=False)

        self.config = config

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        do_lm=False,
        image_token_type_idx=1,
        img=None,
        img_pre_feat=None,
        text_pre_feat=None,
        return_intermediate=False,
        image_only=False,
        text_only=False,
    ):
        if not text_only:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch[imgkey][0]

            raw_image_embeds = self.vit_model(img)

        if image_only:
            return {"image_embeds": raw_image_embeds}

        input_suffix = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{input_suffix}"]
        output_suffix = "_lm" if do_lm else input_suffix
        text_labels = batch[f"text_labels{output_suffix}"]
        text_masks = batch[f"text_masks"]

        subtitle_text_ids = batch[f"subtitle_text_ids"]
        subtitle_text_masks = batch[f"subtitle_text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        subtitle_text_embeds = self.text_transformer.embeddings(input_ids=subtitle_text_ids)

        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        extend_subtitle_text_masks = subtitle_text_masks[:, None, None, :]

        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, attention_mask=extend_text_masks)[0]
        raw_text_embeds = text_embeds

        for layer in self.text_transformer.encoder.layer:
            subtitle_text_embeds = layer(subtitle_text_embeds, attention_mask=extend_subtitle_text_masks)[0]
        raw_subtitle_text_embeds = subtitle_text_embeds

        if text_only:
            return {"text_embeds": raw_text_embeds}

        if return_intermediate:
            ret = {
                "text_embeds": raw_text_embeds,
                "image_embeds": raw_image_embeds,
                "subtitle_text_embeds": raw_subtitle_text_embeds,
            }
            return ret

        text_embeds = self.cross_modal_text_transform(raw_text_embeds)
        subtitle_text_embeds = self.cross_modal_text_transform(raw_subtitle_text_embeds)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        subtitle_text_embeds = subtitle_text_embeds + self.token_type_embeddings(
            torch.full_like(
                subtitle_text_masks,
                image_token_type_idx + len(self.token_type_embeddings.weight.data) // 2,
            )
        )

        image_embeds = self.cross_modal_image_transform(raw_image_embeds)
        image_masks = torch.ones(
            (image_embeds.size(0), image_embeds.size(1)),
            dtype=torch.long,
            device=image_embeds.device,
        )
        # extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)
        extend_image_masks = image_masks.reshape(image_masks.size(0), 1, 1, image_masks.size(1))
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        extend_image_subtitle_masks = torch.cat((extend_subtitle_text_masks, extend_image_masks), dim=-1)
        image_embeds = torch.cat([image_embeds, subtitle_text_embeds], dim=-2)
        x, y = text_embeds, image_embeds

        if self.config["disable_cross_modal_image_layer"]:
            for text_layer in self.cross_modal_text_layers:
                x1 = text_layer(
                    x,
                    y,
                    attention_mask=extend_text_masks,
                    encoder_attention_mask=extend_image_subtitle_masks,
                )
                x = x1[0]
            text_feats, image_feats = x, y
            cls_feats = self.cross_modal_text_pooler(x)
            cls_feats = torch.cat([cls_feats, cls_feats], dim=-1)
        else:
            for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
                x1 = text_layer(
                    x,
                    y,
                    attention_mask=extend_text_masks,
                    encoder_attention_mask=extend_image_subtitle_masks,
                )
                y1 = image_layer(
                    y,
                    x,
                    attention_mask=extend_image_subtitle_masks,
                    encoder_attention_mask=extend_text_masks,
                )
                x, y = x1[0], y1[0]

            text_feats, image_feats = x, y
            cls_feats_text = self.cross_modal_text_pooler(x)
            if self.is_clip or self.is_videoformer:
                cls_feats_image = self.cross_modal_image_pooler(y)
            elif self.is_swin:
                avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
                cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
            cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_embeds": raw_text_embeds,
            "image_embeds": raw_image_embeds,
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        loss_weights = self.hparams.config["loss_names"]
        # pdb.set_trace()
        total_loss = sum([loss_weights[k.split("_")[0]] * v for k, v in output.items() if "loss" in k])
        if self.config["debug"]:
            print([(k, v) for k, v in output.items() if "loss" in k])
            print("total_loss: {}".format(total_loss))
        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        if not self.hparams.config["skip_test_step"]:
            output = self(batch)
        ret = dict()
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
