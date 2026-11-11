import torch
import torch.nn.functional as F
from .attention_couple import AttentionCouple

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


NODE_CLASS_MAPPINGS = {
    "Anima Character": AnimaCharacter,
    "Anima Character Prompts": AnimaCharacterPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima Character": "Anima Character",
    "Anima Character Prompts": "Anima Character Prompts",
}
