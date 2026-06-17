import torch
import torch.nn.functional as F
import copy
import comfy
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

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

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
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode, isolation_factor, cross_region_blend=0.3):
        if mode == "Latent":
            return (model, positive, negative)  # latent coupleの場合は何もしない

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
            cond_norms = F.normalize(cond_means, dim=-1)
            sim_matrix = torch.mm(cond_norms, cond_norms.t())  # [n_regions, n_regions]
            sim_matrix = sim_matrix.clamp(min=0) ** 2  # sharpen: similar stays high, dissimilar drops
            sim_matrix = sim_matrix / sim_matrix.sum(dim=1, keepdim=True)  # normalize rows
            self.cond_similarity = sim_matrix
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

    def make_patch(self, module):           
        def patch(q, k, v, extra_options):
            len_neg, len_pos = self.conditioning_length
            cond_or_uncond = extra_options["cond_or_uncond"]
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0]

            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            # Pad conditionings to max token length instead of truncating
            def pad_conds_to_max(conds):
                max_size = max(c.shape[1] for c in conds)
                result = []
                for c in conds:
                    if c.shape[1] < max_size:
                        pad = torch.zeros(c.shape[0], max_size - c.shape[1], c.shape[2],
                                          device=c.device, dtype=c.dtype)
                        c = torch.cat([c, pad], dim=1)
                    result.append(c)
                return result

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