"""
@author: Rei D. / neeltheninja
@title: Comfy Couple / ComfyEnhancedMultiRegion
@nickname: Danand / Neel
@description: If you want to draw multiple different regions together without blending their features, so you could try to check out this custom node.
"""

from .ComfyEnhancedMultiRegion import NODE_CLASS_MAPPINGS as _MULTI_REGION, NODE_DISPLAY_NAME_MAPPINGS as _MULTI_REGION_NAMES
from .anima_character_prompts import NODE_CLASS_MAPPINGS as _ANIMA_CHARACTER, NODE_DISPLAY_NAME_MAPPINGS as _ANIMA_CHARACTER_NAMES

NODE_CLASS_MAPPINGS = {**_MULTI_REGION, **_ANIMA_CHARACTER}
NODE_DISPLAY_NAME_MAPPINGS = {**_MULTI_REGION_NAMES, **_ANIMA_CHARACTER_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
