from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from utils.unet import UNet3DConditionModel
from utils.pipeline_magictime import MagicTimePipeline
from utils.util import save_videos_grid
from utils.util import load_weights
import torch, json, os, requests
from PIL import Image

import runpod

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
def generate(input):
    values = input["input"]
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
    
    result = f"/content/MagicTime/output.mp4"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})