"""VLM prefix embedding extraction for PI05 policies.

Provides mean-pooled embeddings over image tokens from the VLM prefix,
used for Mahalanobis distance-based OOD detection.
"""

from typing import cast

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


@torch.no_grad()
def embed_prefix_pooled(policy: PI05Policy, batch: dict) -> torch.Tensor:
    """Run a batch through the VLM prefix and return mean-pooled embeddings.

    Only image tokens are included in the pooling; language and state tokens
    are excluded.

    Args:
        batch: Already preprocessed observation dict on device.

    Returns:
        (B, hidden_dim) mean-pooled over image tokens.
    """
    prefix_out, prefix_pad_masks = embed_prefix_pi05(policy, batch)

    mask = prefix_pad_masks.unsqueeze(-1).float()
    pooled = (prefix_out.float() * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled


def embed_prefix_pi05(policy: PI05Policy, batch: dict):
    """PI05: images + language -> PaliGemma prefix forward (4D attention masks).

    Returns prefix_out and a pooling mask covering image tokens only
    (language token positions are masked out).
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    q_dtype = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj.weight.dtype
    prefix_embs = prefix_embs.to(dtype=q_dtype)

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=q_dtype)

    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=cast(torch.LongTensor, prefix_position_ids),
        past_key_values=None,
        inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
        use_cache=False,
    )

    # Pooling mask: image tokens only (prefix layout: [img...][lang...])
    n_lang = lang_tokens.shape[1]
    n_img = prefix_embs.shape[1] - n_lang
    vis_mask = prefix_pad_masks.clone()
    vis_mask[:, n_img:] = False

    return prefix_out, vis_mask
