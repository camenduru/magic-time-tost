from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from utils.unet import UNet3DConditionModel
from utils.pipeline_magictime import MagicTimePipeline
from utils.util import save_videos_grid
from utils.util import load_weights
import torch, json
from PIL import Image

tokenizer    = CLIPTokenizer.from_pretrained("/content/MagicTime/ckpts/Base_Model/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("/content/MagicTime/ckpts/Base_Model/stable-diffusion-v1-5", subfolder="text_encoder").cuda()
vae          = AutoencoderKL.from_pretrained("/content/MagicTime/ckpts/Base_Model/stable-diffusion-v1-5", subfolder="vae").cuda()
unet = UNet3DConditionModel.from_pretrained_2d("/content/MagicTime/ckpts/Base_Model/stable-diffusion-v1-5", subfolder="unet",
unet_additional_kwargs = {
    "use_inflated_groupnorm": True,
    "use_motion_module": True,
    "motion_module_resolutions": [1, 2, 4, 8],
    "motion_module_mid_block": False,
    "motion_module_type": "Vanilla",
    "motion_module_kwargs": {
        "num_attention_heads": 8,
        "num_transformer_block": 1,
        "attention_block_types": ["Temporal_Self", "Temporal_Self"],
        "temporal_position_encoding": True,
        "temporal_position_encoding_max_len": 32,
        "temporal_attention_dim_div": 1,
        "zero_initialize": True
    },
    "noise_scheduler_kwargs": {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "steps_offset": 1,
        "clip_sample": False
    }
}).cuda()

if is_xformers_available():
    unet.enable_xformers_memory_efficient_attention()

pipeline = MagicTimePipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    scheduler=DDIMScheduler(beta_start=0.00085,
                            beta_end=0.012,
                            beta_schedule="linear",
                            steps_offset=1,
                            clip_sample=False)
).to("cuda")

dreambooth = "ToonYou_beta6" # @param ["ToonYou_beta6", "RealisticVisionV60B1_v51VAE", "RcnzCartoon"]

pipeline = load_weights(
    pipeline,
    motion_module_path="/content/MagicTime/ckpts/Base_Model/motion_module/motion_module.ckpt",
    dreambooth_model_path=f"/content/MagicTime/ckpts/DreamBooth/{dreambooth}.safetensors",
    magic_adapter_s_path="/content/MagicTime/ckpts/Magic_Weights/magic_adapter_s/magic_adapter_s.ckpt",
    magic_adapter_t_path="/content/MagicTime/ckpts/Magic_Weights/magic_adapter_t",
    magic_text_encoder_path="/content/MagicTime/ckpts/Magic_Weights/magic_text_encoder",
).to("cuda")

import gradio as gr

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(command):
    values = json.loads(command)
    random_seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    prompt = values['prompt']
    negative_prompt = values['negative_prompt']
    num_inference_steps = values['num_inference_steps']
    guidance_scale = values['guidance_scale']
    width = closestNumber(values['width'], 8)
    height = closestNumber(values['height'], 8)
    video_length = values['video_length']
    sample = pipeline(prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    video_length=video_length,
                    ).videos
    save_videos_grid(sample, f"/content/MagicTime/output.mp4")
    return f"/content/MagicTime/output.mp4"

with gr.Blocks(css=".gradio-container {max-width: 544px !important}", analytics_enabled=False) as demo:
    with gr.Row():
      with gr.Column():
          textbox = gr.Textbox(show_label=False, 
          value="""{"prompt":"An ice cube is melting."}""")
          button = gr.Button()
    with gr.Row(variant="default"):
        output_video = gr.Video(
            show_label=False,
            interactive=False,
            height=512,
            width=512,
            elem_id="output_video",
        )

    button.click(fn=generate, inputs=[textbox], outputs=[output_video])

import os
PORT = int(os.getenv('server_port'))
demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0', server_port=PORT)