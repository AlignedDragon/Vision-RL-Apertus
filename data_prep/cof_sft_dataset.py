"""SFT dataset for Apertus rendered conversations with intra-turn tool-output masking.

Reads the JSONL produced by `cof_sft_parse.py` (rows of
`{"text": <rendered Apertus chat>, "image_paths": [...]}`) and emits per-example
`{input_ids, attention_mask, position_ids, loss_mask}` tensors for verl's SFT
trainer (custom_cls hook in verl/trainer/sft_trainer.py:create_sft_dataset).

Loss-mask policy (only assistant generations train):
- Mask everything before each <|assistant_start|> and the <|assistant_start|>
  token itself (it is the generation cue).
- Train on assistant content: thoughts, tool-call JSON, response text, and the
  surrounding <|inner_*|>, <|tools_*|> special tokens, plus <|assistant_end|>.
- Mask each tool-OUTPUT span: the bracketed `[...]` literal that immediately
  follows a <|tools_suffix|> token (matched via JSON-aware bracket-depth count).
"""

from __future__ import annotations

import bisect
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin


A_START = "<|assistant_start|>"
A_END = "<|assistant_end|>"
T_SUF = "<|tools_suffix|>"


def _bracket_close(text: str, open_idx: int, hi: int) -> int:
    """Return index of the ']' matching the '[' at `open_idx`, bounded by hi.

    Bracket-depth count (not find-next-`]`) so JSON arrays inside the tool
    output don't fool us. IBQ image tokens contain no `[`/`]`, so they can't
    create false depth changes.
    """
    depth = 0
    i = open_idx
    while i < hi:
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return hi - 1


def compute_train_spans(text: str) -> list[tuple[int, int]]:
    """Half-open char spans [lo, hi) where assistant content should train.

    Walks through every <|assistant_start|> ... <|assistant_end|> block,
    starting AFTER the start token (so the start token stays masked) and
    INCLUDING the end token (so the model learns to emit it). A missing
    closing tag (template suppresses it when the turn ends in a tool call)
    runs the span to end-of-text.

    Inside each block, every <|tools_suffix|> followed immediately by a `[`
    starts a tool-output sub-span that gets subtracted from the train span.
    A <|tools_suffix|> NOT followed by `[` is a tool-CALL closer (the model
    emits it) and remains trainable.
    """
    n = len(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        # go through the text and anchor on <|assistant_start|> ... <|assistant_end|> blocks
        s = text.find(A_START, pos)
        if s < 0:
            break
        lo = s + len(A_START)
        e = text.find(A_END, lo)
        hi = e + len(A_END) if e >= 0 else n

        mask_subs: list[tuple[int, int]] = []
        cur = lo
        while True:
            # anchor on <|tools_suffix|> to find the tool outputs delinieated by []
            ts = text.find(T_SUF, cur, hi)
            if ts < 0:
                break
            after = ts + len(T_SUF)
            if after < hi and text[after] == "[":
                cur = _bracket_close(text, after, hi) + 1
                mask_subs.append((after, cur))
            else:
                cur = after

        cursor = lo
        for mlo, mhi in mask_subs:
            if mlo > cursor:
                # include non-tool-output lines only. 
                spans.append((cursor, mlo))
            cursor = max(cursor, mhi)
        if cursor < hi:
            spans.append((cursor, hi))

        pos = hi
    return spans


def _spans_to_loss_mask(
    offset_mapping: list[tuple[int, int]],
    train_spans: list[tuple[int, int]],
    length: int,
) -> torch.Tensor:
    loss_mask = torch.zeros(length, dtype=torch.long)
    if not train_spans:
        return loss_mask
    starts = [s for s, _ in train_spans]
    ends = [e for _, e in train_spans]
    for i, (a, b) in enumerate(offset_mapping):
        idx = bisect.bisect_right(starts, a) - 1
        if idx < 0:
            continue
        # a fits somewhere among starts elements
        if b > a:
            if b <= ends[idx]:
                loss_mask[i] = 1
        else:
            # in case the token is interpreted as having 0 length
            if a < ends[idx]:
                loss_mask[i] = 1
    return loss_mask


class CoFSFTDataset(Dataset):
    """Custom verl SFT dataset for pre-rendered Apertus conversations.

    Args mirror verl's `create_sft_dataset` factory call signature.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: Optional[DictConfig] = None,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ("right", "no_padding"), self.pad_mode
        self.max_length = int(config.get("max_length", 8192))
        self.truncation = config.get("truncation", "right")
        assert self.truncation in ("left", "right", "error"), self.truncation
        self.text_key = config.get("text_key", "text")

        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        frames = []
        for path in parquet_files:
            path = str(path)
            if path.endswith(".jsonl"):
                frames.append(pd.read_json(path, lines=True))
            else:
                frames.append(pd.read_parquet(path, dtype_backend="pyarrow"))
        self.dataframe = pd.concat(frames, ignore_index=True)

        total = len(self.dataframe)
        if max_samples is not None and max_samples > 0 and max_samples < total:
            self.dataframe = self.dataframe.iloc[:max_samples].reset_index(drop=True)
        print(f"CoFSFTDataset: {len(self.dataframe)}/{total} rows (pad_mode={self.pad_mode}, max_length={self.max_length})")

    def __len__(self):
        return len(self.dataframe)

    def _build_one(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.dataframe.iloc[idx][self.text_key]
        # add_special_tokens=False: the rendered text already begins with <s>;
        # otherwise HF prepends a duplicate BOS and shifts every offset.
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        offset_mapping = enc["offset_mapping"]

        train_spans = compute_train_spans(text)
        loss_mask = _spans_to_loss_mask(offset_mapping, train_spans, len(input_ids))

        seq_len = input_ids.shape[0]
        position_ids = torch.arange(seq_len, dtype=torch.long)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        if seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                position_ids = torch.arange(self.max_length, dtype=torch.long)
            else:
                raise ValueError(f"sequence length {seq_len} exceeds max_length {self.max_length}")
            seq_len = self.max_length

        if self.pad_mode == "right":
            if seq_len < self.max_length:
                pad_len = self.max_length - seq_len
                pad_id = self.tokenizer.pad_token_id
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)])
                position_ids = F.pad(position_ids, (0, pad_len), value=0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self.dataframe)
        item = self._build_one(idx)
        if int(item["loss_mask"].sum()) > 0:
            return item
        # All-masked row (typically truncation killed all assistant tokens).
        # Skip forward to avoid a zero-loss microbatch / nan division.
        for k in range(1, min(8, n)):
            j = (idx + k) % n
            alt = self._build_one(j)
            if int(alt["loss_mask"].sum()) > 0:
                print(f"CoFSFTDataset: row {idx} had loss_mask.sum()=0; using row {j} instead")
                return alt
        return item


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data_prep/cof_sft/metadata.jsonl")
    p.add_argument("--checkpoint", default=None, help="Apertus HF checkpoint dir")
    p.add_argument("--config", default="configs/apertus.yaml")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=2)
    p.add_argument("--max-length", type=int, default=32768)
    args = p.parse_args()

    if args.checkpoint is None:
        cfg = OmegaConf.load(args.config)
        ckpt = cfg.model.checkpoint
    else:
        ckpt = args.checkpoint

    print(f"Loading tokenizer from {ckpt}")
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    ds_cfg = OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "max_length": args.max_length,
            "truncation": "right",
            "text_key": "text",
        }
    )
    ds = CoFSFTDataset([args.metadata], tok, ds_cfg, processor=None, max_samples=args.max_samples)
    item = ds[args.idx]
    ids = item["input_ids"]
    lm = item["loss_mask"].bool()
    print(f"seq_len={len(ids)}  trainable_tokens={int(lm.sum())}")
    bos_count = int((ids == tok.bos_token_id).sum())
    print(f"bos_token_id={tok.bos_token_id}  bos_count_in_seq={bos_count}  (expected 1)")
    print("=== TRAINABLE DECODE ===")
    print(tok.decode(ids[lm].tolist()))
    print("=== MASKED PREFIX (first 200 tokens) ===")
    print(tok.decode(ids[~lm][:200].tolist()))
