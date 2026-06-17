import math
import traceback
import torch
from nodes import MAX_RESOLUTION, ConditioningCombine, ConditioningSetMask
from comfy_extras.nodes_mask import MaskComposite, SolidMask
from .attention_couple import AttentionCouple

class ComfyMultiRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "negative": ("CONDITIONING",),
                "orientation": (["horizontal", "vertical"],),
                "num_regions": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "cross_region_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                **{f"positive_{i+1}": ("CONDITIONING",) for i in range(10)},
                **{f"ratio_{i+1}": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}) for i in range(9)},
                **{f"weight_{i+1}": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}) for i in range(10)}
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, model, negative, orientation, num_regions, width, height, isolation_factor, **kwargs):
        try:
            cross_region_blend = kwargs.get("cross_region_blend", 0.3)
            positives = [kwargs.get(f"positive_{i+1}") for i in range(num_regions)]
            ratios = [kwargs.get(f"ratio_{i+1}", 1.0 / num_regions) for i in range(num_regions - 1)]
            weights = [kwargs.get(f"weight_{i+1}", 1.0) for i in range(num_regions)]

            if any(pos is None for pos in positives):
                raise ValueError(f"Expected {num_regions} positive conditionings, but some are missing")

            # Normalize ratios
            ratios.append(max(0, 1.0 - sum(ratios)))  # Ensure non-negative
            total = sum(ratios)
            ratios = [r / total for r in ratios]

            # Create masks for each region
            masks = self.create_masks(ratios, orientation, width, height)

            # Apply masks and weights to positive conditionings
            conditioned_masks = [ConditioningSetMask().append(pos, mask, "default", weight)[0] for pos, mask, weight in zip(positives, masks, weights)]

            # Combine all conditioned masks
            positive_combined = conditioned_masks[0]
            for mask in conditioned_masks[1:]:
                positive_combined = ConditioningCombine().combine(positive_combined, mask)[0]

            return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention", isolation_factor, cross_region_blend)
        except Exception as e:
            print(f"[ComfyMultiRegion] ERROR: {str(e)}")
            traceback.print_exc()
            return None, None, None

    @staticmethod
    def create_masks(ratios, orientation, width, height):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_regions = len(ratios)
        masks = torch.zeros((num_regions, height, width), device=device)

        # Calculate region boundaries in pixels
        main_dim = width if orientation == "horizontal" else height
        boundaries = [0]
        for ratio in ratios[:-1]:
            boundaries.append(boundaries[-1] + math.floor(main_dim * ratio))
        boundaries.append(main_dim)

        # Feather width as fraction of image dimension (~12.5%)
        feather_pixels = max(8, main_dim // 8)
        scale = feather_pixels / 4  # sigmoid steepness: transition ~feather_pixels wide

        if orientation == "horizontal":
            x = torch.arange(width, device=device).float()
            for i in range(num_regions):
                left = boundaries[i]
                right = boundaries[i + 1]
                if num_regions == 1:
                    masks[i] = 1.0
                elif i == 0:
                    masks[i] = 1.0 - torch.sigmoid((x - right) / scale)
                elif i == num_regions - 1:
                    masks[i] = torch.sigmoid((x - left) / scale)
                else:
                    masks[i] = torch.sigmoid((x - left) / scale) * \
                              (1.0 - torch.sigmoid((x - right) / scale))
        else:  # vertical
            y = torch.arange(height, device=device).float()
            scale = feather_pixels / 4
            for i in range(num_regions):
                top = boundaries[i]
                bottom = boundaries[i + 1]
                if num_regions == 1:
                    masks[i] = 1.0
                elif i == 0:
                    masks[i] = 1.0 - torch.sigmoid((y - bottom) / scale)
                elif i == num_regions - 1:
                    masks[i] = torch.sigmoid((y - top) / scale)
                else:
                    masks[i] = torch.sigmoid((y - top) / scale) * \
                              (1.0 - torch.sigmoid((y - bottom) / scale))

        # Ensure sum to 1 (numerical safety)
        mask_sum = masks.sum(dim=0, keepdim=True)
        masks = masks / (mask_sum + 1e-6)

        return masks

NODE_CLASS_MAPPINGS = {
    "Comfy Multi-Region": ComfyMultiRegion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy Multi-Region": "Comfy Multi-Region",
}