import math
import os
import torch
import torch.nn.functional as F
import comfy.model_base
import comfy.patcher_extension
from .attention_couple import AttentionCouple, pad_conds_to_max

# Character areas are drawn on a square canvas in normalized coordinates; the Anima patch resizes the
# masks to the real token grid of the latent at sampling time, so the canvas aspect ratio does not matter.
CANVAS = 256
MAX_CHARACTERS = 6


def box_mask(x, y, width, height):
    mask = torch.zeros(1, CANVAS, CANVAS)
    x0 = max(0, round((x - width / 2) * CANVAS))
    x1 = min(CANVAS, round((x + width / 2) * CANVAS))
    y0 = max(0, round((y - height / 2) * CANVAS))
    y1 = min(CANVAS, round((y + height / 2) * CANVAS))
    mask[:, y0:y1, x0:x1] = 1.0
    return mask


def canvas_mask(mask):
    # MASK input: [B, H, W] or [H, W], any size -> [1, CANVAS, CANVAS]
    if mask.ndim == 3:
        mask = mask[:1]
    else:
        mask = mask.unsqueeze(0)
    mask = F.interpolate(mask.unsqueeze(0).float(), size=(CANVAS, CANVAS), mode="bilinear", align_corners=False)
    return mask[0].clamp(0.0, 1.0)


class AnimaCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "The prompt of this character only (looks, outfit, expression)."}),
                "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Center of the character's area, 0 = left edge, 1 = right edge."}),
                "y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Center of the character's area, 0 = top edge, 1 = bottom edge."}),
                "width": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Width of the area as a fraction of the image width."}),
                "height": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Height of the area as a fraction of the image height."}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of this character where its area overlaps another one."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional painted area for this character. Replaces x/y/width/height when connected."}),
            },
        }

    RETURN_TYPES = ("ANIMA_CHARACTER",)
    RETURN_NAMES = ("character",)
    FUNCTION = "make"
    CATEGORY = "loaders/anima"

    def make(self, conditioning, x, y, width, height, weight, mask=None):
        area = canvas_mask(mask) if mask is not None else box_mask(x, y, width, height)
        return ({"conditioning": conditioning, "mask": area, "weight": weight},)


class AnimaCharacterPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base": ("CONDITIONING", {"tooltip": "Prompt for the whole scene: composition, interaction, background, quality tags. Used where no character is placed, and mixed into the characters by base_weight."}),
                "negative": ("CONDITIONING",),
                "character_1": ("ANIMA_CHARACTER",),
                "base_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the base prompt is mixed into the character areas (0 = characters only see their own prompt). Everything in the base prompt then reaches every character, so keep character attributes out of it."}),
                "mask_feather": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1, "tooltip": "Soften the edges of the areas over this many tokens (1 token = 16 px)."}),
                "isolation_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Block self-attention between areas (0 = off). Raise only if attributes leak between characters; high values cut limbs that cross into another area."}),
            },
            "optional": {
                **{f"character_{i}": ("ANIMA_CHARACTER",) for i in range(2, MAX_CHARACTERS + 1)},
                "isolation_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of the sampling steps that run without isolation."}),
                "isolation_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Isolation stops at this fraction of the sampling steps."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "loaders/anima"

    def apply(self, model, base, negative, base_weight, mask_feather, isolation_factor,
              isolation_start=0.0, isolation_end=1.0, **characters):
        chars = [characters[f"character_{i}"] for i in range(1, MAX_CHARACTERS + 1) if characters.get(f"character_{i}") is not None]

        # The base prompt owns everything no character covers
        coverage = torch.stack([c["mask"] for c in chars]).sum(dim=0)
        background = (1.0 - coverage).clamp(0.0, 1.0)

        positive = [[base[0][0], dict(base[0][1], mask=background, mask_strength=1.0)]]
        for c in chars:
            cond = c["conditioning"][0]
            positive.append([cond[0], dict(cond[1], mask=c["mask"], mask_strength=c["weight"])])

        return AttentionCouple().attention_couple(
            model, positive, negative, "Attention", isolation_factor,
            cross_region_blend=0.0, model_type="Anima",
            isolation_start=isolation_start, isolation_end=isolation_end,
            base_positive=base if base_weight > 0 else None, base_weight=base_weight, mask_feather=mask_feather)


def derive_masks(affinity, grid, sharpness, feather):
    """affinity: [b, segments, N] attention mass from image tokens to each prompt (segment 0 = base).
    Returns soft masks [b, segments, N] that sum to 1 over the segments."""
    b, segments, n = affinity.shape
    if grid is not None and grid[0] * grid[1] == n:
        # light denoise of the raw maps before comparing them
        affinity = F.avg_pool2d(affinity.reshape(b * segments, 1, *grid), 3, stride=1, padding=1, count_include_pad=False).reshape(b, segments, n)
    # each prompt is compared with its own average, so long or generic prompts do not win everywhere
    relative = affinity / affinity.mean(dim=2, keepdim=True).clamp(min=1e-12)
    masks = torch.softmax(sharpness * torch.log(relative.clamp(min=1e-6)), dim=1)
    if feather > 0 and grid is not None and grid[0] * grid[1] == n:
        masks = F.avg_pool2d(masks.reshape(b * segments, 1, *grid), 2 * feather + 1, stride=1, padding=feather, count_include_pad=False).reshape(b, segments, n)
        masks = masks / masks.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return masks


PALETTE = [(128, 128, 128), (230, 60, 60), (60, 120, 230), (60, 190, 90), (230, 180, 40), (170, 80, 210), (40, 200, 200)]


def save_layout_preview(masks, grid):
    try:
        import folder_paths
        from PIL import Image
        colors = torch.tensor(PALETTE[:masks.shape[1]], dtype=torch.float32)  # base = gray, characters in palette order
        rgb = torch.einsum("sn,sc->nc", masks[0].float().cpu(), colors).reshape(*grid, 3).clamp(0, 255).byte()
        image = Image.fromarray(rgb.numpy()).resize((grid[1] * 16, grid[0] * 16), Image.NEAREST)
        out_dir = folder_paths.get_output_directory()
        i = 0
        while os.path.exists(os.path.join(out_dir, f"anima_auto_layout_{i:05}.png")):
            i += 1
        path = os.path.join(out_dir, f"anima_auto_layout_{i:05}.png")
        image.save(path)
        print(f"[Anima Auto Character Prompts] layout preview saved to {path}")
    except Exception as e:
        print(f"[Anima Auto Character Prompts] could not save the layout preview: {e}")


class AutoLayout:
    """Finds where each character is from the model's own cross-attention during the first steps
    (all prompts combined), then gives every character only its own prompt in its area."""

    def __init__(self, base_model, base, characters, negative, layout_end, sharpness, base_weight, feather, preview):
        self.base_model = base_model
        self.prompts = [self.text(base)] + [self.text(c) for c in characters]  # segment 0 = base
        self.negative = self.text(negative)
        self.layout_end = layout_end
        self.sharpness = sharpness
        self.base_weight = base_weight
        self.feather = feather
        self.preview = preview
        self.n_blocks = len(base_model.diffusion_model.blocks)
        self.contexts = None
        self.contexts_key = None
        self.run_key = None
        self.affinity = None  # summed attention mass per prompt during the layout steps, [b, segments, N]
        self.masks = None

    @staticmethod
    def text(conditioning):
        cond = conditioning[0]
        return cond[0], cond[1].get("t5xxl_ids"), cond[1].get("t5xxl_weights")

    def encode(self, prompt, device, dtype):
        emb, ids, weights = prompt
        emb = emb.to(device=device, dtype=dtype)
        if ids is None:
            return emb, emb.shape[1]
        if weights is not None:
            weights = weights.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype)
        # same preprocessing as comfy.model_base.Anima.extra_conds (Qwen3 -> LLM adapter, padded to 512)
        return self.base_model.diffusion_model.preprocess_text_embeds(emb, ids.unsqueeze(0).to(device), t5xxl_weights=weights), ids.shape[-1]

    def get_contexts(self, device, dtype):
        if self.contexts_key != (device, dtype):
            encoded = [self.encode(p, device, dtype) for p in self.prompts]
            bounds, start = [], 0
            for _, length in encoded:
                bounds.append((start, start + length))
                start += length
            joint = torch.cat([ctx[:, :length] for ctx, length in encoded], dim=1)  # every prompt, padding removed
            if joint.shape[1] < 512:
                joint = F.pad(joint, (0, 0, 0, 512 - joint.shape[1]))
            self.contexts = {
                "joint": joint,
                "bounds": bounds,
                "regional": torch.cat(pad_conds_to_max([ctx for ctx, _ in encoded]), dim=0),
                "negative": self.encode(self.negative, device, dtype)[0],
            }
            self.contexts_key = (device, dtype)
        return self.contexts

    def wrapper(self, executor, x, *args, **kwargs):
        options = kwargs.get("transformer_options")
        if options is not None:
            grid = (math.ceil(x.shape[-2] / 2), math.ceil(x.shape[-1] / 2))
            sigmas, current = options.get("sample_sigmas"), options.get("sigmas")
            if sigmas is not None and current is not None and len(sigmas) > 1:
                step = (sigmas.to(current.device).float() - current.flatten()[0].float()).abs().argmin().item()
                progress = step / (len(sigmas) - 1)
                run = (grid, len(sigmas), round(float(sigmas[0]), 4))
            else:
                progress, run = 0.0, (grid,)
            if run != self.run_key:  # new sampling run (e.g. the hires pass): find the layout again
                self.run_key = run
                self.affinity = None
                self.masks = None
            options["anima_auto_grid"] = grid
            options["anima_auto_layout"] = progress < self.layout_end
        return executor(x, *args, **kwargs)

    def get_masks(self, grid):
        if self.masks is None and self.affinity is not None:
            self.masks = derive_masks(self.affinity, grid, self.sharpness, self.feather)
            if self.preview:
                save_layout_preview(self.masks, grid)
        return self.masks

    def joint_attention(self, module, x, ctx, rope_emb, transformer_options, measure):
        q, k, v = module.compute_qkv(x, ctx["joint"], rope_emb=rope_emb, transformer_options=transformer_options)
        b, n = x.shape[:2]
        if not measure:
            attn_kwargs = {"transformer_options": transformer_options}
            if hasattr(module, "comfy_attention"):
                attn_kwargs["preferred_attention"] = module.comfy_attention
            attn = module.attn_op(q, k.repeat(b, 1, 1, 1).to(q.dtype), v.repeat(b, 1, 1, 1).to(q.dtype), **attn_kwargs)
            return module.output_dropout(module.output_proj(attn))
        # explicit attention so the weights can be read: [b, heads, N, L]
        probs = (torch.einsum("bnhd,lhd->bhnl", q.float(), k[0].float()) * q.shape[-1] ** -0.5).softmax(dim=-1)
        attn = torch.einsum("bhnl,lhd->bnhd", probs, v[0].float()).reshape(b, n, -1).to(q.dtype)
        mass = torch.stack([probs[..., s:e].sum(dim=-1) for s, e in ctx["bounds"]], dim=1).mean(dim=2)  # [b, segments, N]
        self.affinity = mass if self.affinity is None else self.affinity + mass
        return module.output_dropout(module.output_proj(attn))

    def regional_attention(self, module, x, ctx, masks, rope_emb, transformer_options):
        regional = ctx["regional"]
        segments = regional.shape[0]
        b, n = x.shape[:2]
        q, k, v = module.compute_qkv(x, regional, rope_emb=rope_emb, transformer_options=transformer_options)
        attn_kwargs = {"transformer_options": transformer_options}
        if hasattr(module, "comfy_attention"):
            attn_kwargs["preferred_attention"] = module.comfy_attention
        attn = module.attn_op(q.repeat(segments, 1, 1, 1), k.repeat_interleave(b, dim=0).to(q.dtype),
                              v.repeat_interleave(b, dim=0).to(q.dtype), **attn_kwargs).view(segments, b, n, -1)
        out = (attn * masks.permute(1, 0, 2).unsqueeze(-1).to(attn.dtype)).sum(dim=0)
        if self.base_weight > 0:
            out = (1.0 - self.base_weight) * out + self.base_weight * attn[0]
        return module.output_dropout(module.output_proj(out))

    def make_patch(self, module):
        def forward(x, context=None, rope_emb=None, transformer_options={}):
            cond_or_uncond = transformer_options.get("cond_or_uncond")
            if not cond_or_uncond or x.shape[0] % len(cond_or_uncond) != 0:
                return type(module).forward(module, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            get_dtype = getattr(self.base_model, "get_dtype_inference", None)
            ctx = self.get_contexts(x.device, get_dtype() if get_dtype is not None else x.dtype)
            layout = transformer_options.get("anima_auto_layout", False)
            block = transformer_options.get("block_index", 0)
            # read the layout from the middle blocks, where the maps are the most informative
            measure = layout and self.n_blocks // 4 <= block < max(self.n_blocks // 4 + 1, (3 * self.n_blocks) // 4)
            out = []
            for x_chunk, c in zip(x.chunk(len(cond_or_uncond), dim=0), cond_or_uncond):
                if c != 0:  # negative: plain cross-attention with the negative prompt
                    out.append(type(module).forward(module, x_chunk, ctx["negative"].repeat(x_chunk.shape[0], 1, 1), rope_emb=rope_emb, transformer_options=transformer_options))
                    continue
                masks = None if layout else self.get_masks(transformer_options.get("anima_auto_grid"))
                if masks is None:
                    out.append(self.joint_attention(module, x_chunk, ctx, rope_emb, transformer_options, measure))
                else:
                    out.append(self.regional_attention(module, x_chunk, ctx, masks.to(x.device), rope_emb, transformer_options))
            return torch.cat(out, dim=0)
        return forward


class AnimaAutoCharacterPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base": ("CONDITIONING", {"tooltip": "Prompt for the whole scene: composition, how the characters interact, background, quality tags. Also used for everything that is not a character."}),
                "negative": ("CONDITIONING",),
                "character_1": ("CONDITIONING", {"tooltip": "Prompt of one character only (looks, outfit, expression). No position needed."}),
                "layout_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of the steps that run with all prompts combined while the node finds where each character is. After that every character only gets its own prompt in its area. Higher = better found areas but more mixing in the first steps."}),
                "sharpness": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 16.0, "step": 0.5, "tooltip": "How hard the found areas are separated. Low = soft, overlapping areas; high = sharp borders."}),
                "base_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the base prompt is mixed into the character areas after the layout steps."}),
                "mask_feather": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1, "tooltip": "Smooth the found areas over this many tokens (1 token = 16 px)."}),
                "save_layout_preview": ("BOOLEAN", {"default": False, "tooltip": "Save a picture of the found areas to the output folder (gray = base, then red, blue, green, yellow, purple, cyan for characters 1-6)."}),
            },
            "optional": {f"character_{i}": ("CONDITIONING",) for i in range(2, MAX_CHARACTERS + 1)},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "loaders/anima"

    def apply(self, model, base, negative, layout_end, sharpness, base_weight, mask_feather, save_layout_preview=False, **characters):
        anima_class = getattr(comfy.model_base, "Anima", None)
        if anima_class is None or not isinstance(model.model, anima_class):
            raise ValueError("Anima Auto Character Prompts only works with an Anima model")
        chars = [characters[f"character_{i}"] for i in range(1, MAX_CHARACTERS + 1) if characters.get(f"character_{i}") is not None]

        new_model = model.clone()
        layout = AutoLayout(new_model.model, base, chars, negative, layout_end, sharpness, base_weight, mask_feather, save_layout_preview)
        new_model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "anima_auto_layout", layout.wrapper)
        for i, block in enumerate(new_model.model.diffusion_model.blocks):
            new_model.add_object_patch(f"diffusion_model.blocks.{i}.cross_attn.forward", layout.make_patch(block.cross_attn))
        return (new_model, base, negative)


NODE_CLASS_MAPPINGS = {
    "Anima Character": AnimaCharacter,
    "Anima Character Prompts": AnimaCharacterPrompts,
    "Anima Auto Character Prompts": AnimaAutoCharacterPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima Character": "Anima Character",
    "Anima Character Prompts": "Anima Character Prompts",
    "Anima Auto Character Prompts": "Anima Auto Character Prompts",
}
