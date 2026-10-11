# Comfy EnhancedMultiRegion
![shapes](docs/images/image.png)
## What this node does: 

This is simple custom node for [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) which helps to generate images with _regional prompting_ way easier.

If you want to draw different regions together without blending their features, check out this custom node.

| ⭕ with ComfyEnhancedMultiRegion (I mistakenly kept "3D rendered cone" in all of the prompts XD)| ❌ without ComfyEnhancedMultiRegion |
| --- | --- |
| ![shapes](docs/images/ComfyUI_postColor_00295_.png) | ![shapes](docs/images/ComfyUI_postColor_00296_.png) |
| Prompt 1: ((blue cube)), 3D rendered cone,  white background, (far left)
Prompt 2: ((yellow sphere)), 3D rendered cone,  red background 
Prompt 3: ((purple straight cone)), 3D rendered cone, black background, (far right) | _Single Prompt_: ((blue cube)), 3D rendered cone,  white background, (far left), ((yellow sphere)), 3D rendered cone,  red background, ((purple straight cone)), 3D rendered cone, black background, (far right) |

This is a fork of [**Danand/ComfyUI-ComfyCouple**](https://github.com/Danand/ComfyUI-ComfyCouple), implementing FP16 support and upto 10 different regions in a single image.

## Installation

1. Change directory to custom nodes of **ComfyUI**:

   ```bash
   cd ~/ComfyUI/custom_nodes
   ```

2. Clone this repo here:

   ```bash
   git clone https://github.com/neeltheninja/ComfyUI-ComfyEnhancedMultiRegion.git
   ```

3. Restart **ComfyUI**.

## Usage

1. Right click in workflow.
2. Choose node: **loaders → ComfyEnhancedMultiRegion**
3. Connect inputs (as many positive conditionings as you need), a single negative conditioning, and the clips. Convert the width and height Widgets to Inputs and feed the image latent size to the node. (Refer to my workflow).
4. Connect outputs

Example workflow is [here](workflows/MultiRegionalWorkflow.json).

### Model type (SDXL / Anima)

The `model_type` input (last widget on the node) selects which architecture the attention patch targets:

- **SDXL (Illustrious / Pony / SDXL Base)**: default, same behavior as before (SD1.5 is still detected automatically).
- **Anima**: for [Anima](https://huggingface.co/circlestone-labs/Anima) (Cosmos-Predict2 DiT). Encode the region prompts with the Anima text encoder (Qwen3 0.6B) as in a normal Anima workflow; the node passes them through the model's LLM adapter internally. The region layout is read from the real latent at sampling time, so `width`/`height` only define the proportions of the regions (they do not have to match the image size). Preferably use multiples of 16 (one Anima token covers 16×16 pixels).

If `model_type` does not match the connected model, the node prints an error to the console and outputs nothing.

**Reducing leakage between regions (Anima)**

Anima's self-attention is global, so text bound to one region (e.g. `red eyes`) can leak into the others. Two inputs control this:

- `isolation_factor`: strength of the self-attention block between regions (0 = off, 1 = fully blocked; attention towards other regions is scaled by `(1 - isolation_factor)²`). Try 0.8–0.95 if attributes leak; very high values can make the regions look disconnected (seams, mismatched lighting).
- `cross_region_blend`: mixes the output of regions whose prompts are similar. Prompts that share most tags (quality tags, `2girls`, ...) count as similar, so this leaks by design. Set it to 0 when attributes leak.

On SDXL, `isolation_factor` only sharpens soft masks and has no effect on the hard masks this node generates.



## Credits

- [**@laksjdjf**](https://github.com/laksjdjf) – [Attention-Couple](https://github.com/laksjdjf/attention-couple-ComfyUI) and [**@Danand**](https://github.com/Danand) – [Comfy-Couple](https://github.com/Danand/ComfyUI-ComfyCouple).
