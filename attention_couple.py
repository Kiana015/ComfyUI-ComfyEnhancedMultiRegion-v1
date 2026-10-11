import torch
import torch.nn.functional as F
import copy
import math
import comfy
import comfy.model_base
from comfy.ldm.modules.attention import optimized_attention

def get_masks_from_q(masks, q, original_shape):
    if original_shape[2] * original_shape[3] == q.shape[1]:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == q.shape[1]:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == q.shape[1]:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    ret_masks = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            size = (original_shape[2] // down_sample_rate, original_shape[3] // down_sample_rate)
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            mask_downsample = mask_downsample.view(1,-1, 1).repeat(q.shape[0], 1, q.shape[2])
            ret_masks.append(mask_downsample)
        else:  # coupling処理なしの場合
            ret_masks.append(torch.ones_like(q))

    ret_masks = torch.cat(ret_masks, dim=0)
    return ret_masks

def pad_conds_to_max(conds):
    # Pad conditionings to max token length instead of truncating
    max_size = max(c.shape[1] for c in conds)
    result = []
    for c in conds:
        if c.shape[1] < max_size:
            pad = torch.zeros(c.shape[0], max_size - c.shape[1], c.shape[2],
                              device=c.device, dtype=c.dtype)
            c = torch.cat([c, pad], dim=1)
        result.append(c)
    return result

def compute_cond_similarity(cond_means):
    # cond_means: [n_regions, dim]
    cond_norms = F.normalize(cond_means.float(), dim=-1)
    sim_matrix = torch.mm(cond_norms, cond_norms.t())  # [n_regions, n_regions]
    sim_matrix = sim_matrix.clamp(min=0) ** 2  # sharpen: similar stays high, dissimilar drops
    sim_matrix = sim_matrix / sim_matrix.sum(dim=1, keepdim=True)  # normalize rows
    return sim_matrix

def anima_token_grid(n_tokens, mask_h, mask_w):
    # Anima: VAE downscale 8x, then 2x2 patches -> one token per 16x16 pixels
    h = math.ceil(math.ceil(mask_h / 8) / 2)
    w = math.ceil(math.ceil(mask_w / 8) / 2)
    if h * w == n_tokens:
        return h, w
    # Resolution differs from the node's width/height (e.g. hires pass): pick the
    # factorization of n_tokens closest to the mask aspect ratio
    target = math.log(mask_h / mask_w)
    best = (1, n_tokens)
    for d in range(1, math.isqrt(n_tokens) + 1):
        if n_tokens % d == 0:
            for hh, ww in ((d, n_tokens // d), (n_tokens // d, d)):
                if abs(math.log(hh / ww) - target) < abs(math.log(best[0] / best[1]) - target):
                    best = (hh, ww)
    return best

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

MODEL_TYPES = ["SDXL (Illustrious / Pony / SDXL Base)", "Anima"]

class AttentionCouple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "cross_region_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model_type": (MODEL_TYPES,),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode, isolation_factor, cross_region_blend=0.3, model_type=MODEL_TYPES[0]):
        if mode == "Latent":
            return (model, positive, negative)  # latent coupleの場合は何もしない

        anima_class = getattr(comfy.model_base, "Anima", None)
        is_anima = anima_class is not None and isinstance(model.model, anima_class)
        if model_type == "Anima":
            if not is_anima:
                raise ValueError("model_type is set to Anima, but the connected model is not an Anima model")
            return self.attention_couple_anima(model, positive, negative, isolation_factor, cross_region_blend)
        if is_anima:
            raise ValueError("The connected model is Anima, set model_type to Anima")

        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.isolation_factor = isolation_factor
        self.cross_region_blend = cross_region_blend

        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)

        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        # maskとcondをリストに格納する
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            if len(conditions) != 1:
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0)  # 合計が1になるように正規化(他が0の場合mask_strengthの効果がなくなる)
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                del conditions[0][1]["mask"]  # latent coupleの無効化のため
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
        self.conditioning_length = (len(new_negative), len(new_positive))

        # Pre-compute cross-region conditioning similarity matrix
        pos_conds = self.negative_positive_conds[1]
        if len(pos_conds) > 1 and cross_region_blend > 0:
            cond_means = torch.stack([c.mean(dim=(0, 1)) for c in pos_conds])  # [n_regions, dim]
            self.cond_similarity = compute_cond_similarity(cond_means).to(pos_conds[0].dtype)
        else:
            self.cond_similarity = None

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        if not self.sdxl:
            for id in [1,2,4,5,7,8]:  # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]:  # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))
        else:
            for id in [4,5,7,8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))

        return (new_model, [new_positive[0]], [new_negative[0]])  # pool outputは・・・後回し

    def attention_couple_anima(self, model, positive, negative, isolation_factor, cross_region_blend):
        # Anima (Cosmos-Predict2 DiT) has no patches_replace["attn2"] hook, so the
        # forward of every block's cross_attn is replaced through an object patch.
        self.isolation_factor = isolation_factor
        self.cross_region_blend = cross_region_blend

        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)

        # [negative, positive]: (pixel masks [n, 1, H, W] or None, [(qwen3 embeds, t5xxl_ids, t5xxl_weights)])
        self.anima_regions = []
        for conditions in [new_negative, new_positive]:
            texts = [(cond[0], cond[1].get("t5xxl_ids"), cond[1].get("t5xxl_weights")) for cond in conditions]
            if len(conditions) != 1:
                masks = []
                for cond in conditions:
                    mask = cond[1]["mask"].float().cpu() * cond[1]["mask_strength"]
                    masks.append(mask if mask.ndim == 3 else mask.unsqueeze(0))
                mask_norm = torch.stack(masks)
                total = mask_norm.sum(dim=0, keepdim=True)
                # pixels not covered by any region attend to all regions equally instead of becoming NaN
                mask_norm = torch.where(total > 0, mask_norm / total.clamp(min=1e-6), 1.0 / len(conditions))
                del conditions[0][1]["mask"]  # latent coupleの無効化のため
                del conditions[0][1]["mask_strength"]
            else:
                mask_norm = None
            self.anima_regions.append((mask_norm, texts))

        # The text adapter has to run on the sampling device, so contexts are built lazily
        self.anima_contexts = None
        self.anima_contexts_key = None
        self.anima_token_masks = {}

        new_model = model.clone()
        base_model = new_model.model
        # Self-attention is global in a DiT, so text bound to one region leaks into the others through it.
        # isolation_factor biases self-attention against tokens of other regions (0 = off).
        isolate_self_attn = isolation_factor > 0 and self.anima_regions[1][0] is not None
        for i, block in enumerate(base_model.diffusion_model.blocks):
            new_model.add_object_patch(f"diffusion_model.blocks.{i}.cross_attn.forward", self.make_anima_patch(block.cross_attn, base_model))
            if isolate_self_attn:
                new_model.add_object_patch(f"diffusion_model.blocks.{i}.self_attn.forward", self.make_anima_self_attn_patch(block.self_attn))

        return (new_model, [new_positive[0]], [new_negative[0]])

    def get_anima_contexts(self, base_model, device, dtype):
        key = (device, dtype)
        if self.anima_contexts_key == key:
            return self.anima_contexts

        diffusion_model = base_model.diffusion_model
        contexts = []
        valid_lengths = []
        for _, texts in self.anima_regions:
            ctxs = []
            lengths = []
            for cond, ids, weights in texts:
                cond = cond.to(device=device, dtype=dtype)
                if ids is not None:
                    # Same preprocessing as comfy.model_base.Anima.extra_conds (Qwen3 -> LLM adapter)
                    if weights is not None:
                        weights = weights.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype)
                    cond = diffusion_model.preprocess_text_embeds(cond, ids.unsqueeze(0).to(device), t5xxl_weights=weights)
                    lengths.append(ids.shape[-1])
                else:
                    lengths.append(cond.shape[1])
                ctxs.append(cond)
            contexts.append(torch.cat(pad_conds_to_max(ctxs), dim=0))
            valid_lengths.append(lengths)

        pos_ctx = contexts[1]
        if pos_ctx.shape[0] > 1 and self.cross_region_blend > 0:
            cond_means = torch.stack([pos_ctx[j, :max(1, valid_lengths[1][j])].float().mean(dim=0) for j in range(pos_ctx.shape[0])])
            similarity = compute_cond_similarity(cond_means)
        else:
            similarity = None

        self.anima_contexts = (contexts, similarity)
        self.anima_contexts_key = key
        return self.anima_contexts

    def get_anima_token_masks(self, index, n_tokens, device, dtype):
        key = (index, n_tokens, device, dtype)
        if key not in self.anima_token_masks:
            masks = self.anima_regions[index][0]
            h, w = anima_token_grid(n_tokens, masks.shape[-2], masks.shape[-1])
            token_masks = F.interpolate(masks, size=(h, w), mode="nearest").view(masks.shape[0], h * w)
            token_masks = self.sharpen_masks(token_masks, self.isolation_factor)
            self.anima_token_masks[key] = token_masks.to(device=device, dtype=dtype)
        return self.anima_token_masks[key]

    def get_anima_self_attn_bias(self, n_tokens, device, dtype):
        key = ("self_attn", n_tokens, device, dtype)
        if key not in self.anima_token_masks:
            token_masks = self.get_anima_token_masks(1, n_tokens, device, torch.float32)
            top, ids = token_masks.max(dim=0)
            covered = top > 0.999  # tokens not owned by exactly one region stay unrestricted
            blocked = (ids[:, None] != ids[None, :]) & covered[:, None] & covered[None, :]
            # attention weight towards other regions is scaled by (1 - isolation_factor)^2
            strength = 2.0 * math.log(max(1.0 - self.isolation_factor, 1e-4))
            bias = torch.where(blocked, strength, 0.0).to(dtype)
            self.anima_token_masks[key] = bias[None, None]
        return self.anima_token_masks[key]

    def make_anima_self_attn_patch(self, module):
        def forward(x, context=None, rope_emb=None, transformer_options={}):
            if context is not None:
                return type(module).forward(module, x, context, rope_emb=rope_emb, transformer_options=transformer_options)

            # Applied to the negative pass too (it uses the positive region layout): otherwise the negative
            # prediction of a region would depend on the other regions' latent and CFG would amplify that.
            q, k, v = module.compute_qkv(x, None, rope_emb=rope_emb, transformer_options=transformer_options)  # [b, N, heads, dim_head]
            bias = self.get_anima_self_attn_bias(x.shape[1], q.device, q.dtype)
            attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias)
            attn = attn.transpose(1, 2).reshape(*x.shape[:2], -1)
            return module.output_dropout(module.output_proj(attn))
        return forward

    def make_anima_patch(self, module, base_model):
        def forward(x, context=None, rope_emb=None, transformer_options={}):
            cond_or_uncond = transformer_options.get("cond_or_uncond")
            if not cond_or_uncond or x.shape[0] % len(cond_or_uncond) != 0:
                return type(module).forward(module, x, context, rope_emb=rope_emb, transformer_options=transformer_options)

            get_dtype = getattr(base_model, "get_dtype_inference", None)
            ctx_dtype = get_dtype() if get_dtype is not None else x.dtype
            contexts, similarity = self.get_anima_contexts(base_model, x.device, ctx_dtype)

            n_tokens = x.shape[1]
            out = []
            for x_chunk, c in zip(x.chunk(len(cond_or_uncond), dim=0), cond_or_uncond):
                index = 1 if c == 0 else 0  # cond_or_uncond: 0 = positive, 1 = negative
                ctx = contexts[index]
                regions = ctx.shape[0]
                b = x_chunk.shape[0]

                # q: [b, N, heads, dim_head], k/v: [regions, L, heads, dim_head]
                q, k, v = module.compute_qkv(x_chunk, ctx, rope_emb=rope_emb, transformer_options=transformer_options)
                q = q.repeat(regions, 1, 1, 1)
                k = k.repeat_interleave(b, dim=0).to(q.dtype)
                v = v.repeat_interleave(b, dim=0).to(q.dtype)

                attn_kwargs = {"transformer_options": transformer_options}
                if hasattr(module, "comfy_attention"):
                    attn_kwargs["preferred_attention"] = module.comfy_attention
                attn = module.attn_op(q, k, v, **attn_kwargs)  # [regions * b, N, heads * dim_head]
                attn = attn.view(regions, b, n_tokens, -1)

                # Cross-region similarity blending (only for conditional path)
                if similarity is not None and c == 0:
                    blended = torch.einsum('rj,jbsd->rbsd', similarity.to(device=attn.device, dtype=attn.dtype), attn)
                    attn = (1.0 - self.cross_region_blend) * attn + self.cross_region_blend * blended

                if self.anima_regions[index][0] is not None:
                    masks = self.get_anima_token_masks(index, n_tokens, attn.device, attn.dtype)
                    attn = (attn * masks.view(regions, 1, n_tokens, 1)).sum(dim=0)
                else:
                    attn = attn.sum(dim=0)

                # output_proj is linear, so it is applied once after the masked sum
                out.append(module.output_dropout(module.output_proj(attn)))

            return torch.cat(out, dim=0)
        return forward

    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            len_neg, len_pos = self.conditioning_length
            cond_or_uncond = extra_options["cond_or_uncond"]
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0]

            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            padded_neg = pad_conds_to_max(self.negative_positive_conds[0])
            padded_pos = pad_conds_to_max(self.negative_positive_conds[1])

            context_uncond = torch.cat(padded_neg, dim=0)
            context_cond = torch.cat(padded_pos, dim=0)

            k_uncond = module.to_k(context_uncond)
            k_cond = module.to_k(context_cond)
            v_uncond = module.to_v(context_uncond)
            v_cond = module.to_v(context_cond)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:
                    masks = masks_cond
                    k_r = k_cond
                    v_r = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k_r = k_uncond
                    v_r = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k_rep = torch.cat([k_r[j].unsqueeze(0).repeat(b, 1, 1) for j in range(length)], dim=0)
                v_rep = torch.cat([v_r[j].unsqueeze(0).repeat(b, 1, 1) for j in range(length)], dim=0)

                # Convert all tensors to the same dtype as q_target
                k_rep = k_rep.to(dtype=q_target.dtype)
                v_rep = v_rep.to(dtype=q_target.dtype)
                masks = masks.to(dtype=q_target.dtype)

                # Apply sharpened masks based on isolation factor
                sharpened_masks = self.sharpen_masks(masks, self.isolation_factor)
                
                # Regional attention
                qkv_regional = optimized_attention(q_target, k_rep, v_rep, extra_options["n_heads"])

                # Cross-region similarity blending (only for conditional path)
                if self.cond_similarity is not None and c == 0:
                    out_dim = qkv_regional.shape[-1]
                    seq_len = qkv_regional.shape[1]
                    qkv_per_region = qkv_regional.view(length, b, seq_len, out_dim)

                    sim = self.cond_similarity.to(dtype=qkv_per_region.dtype, device=qkv_per_region.device)
                    blended = torch.einsum('rj,jbsd->rbsd', sim, qkv_per_region)

                    blend = self.cross_region_blend
                    qkv_per_region = (1.0 - blend) * qkv_per_region + blend * blended
                    qkv_regional = qkv_per_region.view(length * b, seq_len, out_dim)

                qkv_regional = qkv_regional * sharpened_masks
                qkv = qkv_regional.view(length, b, -1, module.heads * module.dim_head).sum(dim=0)

                out.append(qkv)

            out = torch.cat(out, dim=0)
            return out
        return patch

    def sharpen_masks(self, masks, isolation_factor):
        # Convert isolation_factor to a tensor with the same device and dtype as masks
        isolation_factor_tensor = torch.tensor(isolation_factor, device=masks.device, dtype=masks.dtype)
        
        # Create a sharper transition based on the isolation factor
        sharpened = torch.pow(masks, torch.exp(isolation_factor_tensor))
        
        # Normalize the sharpened masks
        sharpened = sharpened / (sharpened.sum(dim=0, keepdim=True) + 1e-6)
        
        return sharpened

NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}