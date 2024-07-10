from transformers import CLIPTokenizer, CLIPTextModel
import torch
import os
path = './fixed_prompt_embeds_8views'
os.makedirs(path, exist_ok=True)
pretrained_model_name_or_path = '/home/xiaoxiawu/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-unclip/snapshots/e99f66a92bdcd1b0fb0d4b6a9b81b3b37d8bea44/'
weight_dtype = torch.float16
device = torch.device("cuda")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder')
text_encoder = text_encoder.to(weight_dtype).cuda() 

VIEWS = ["front", "left", "front_left", "back", "right", "front_right",  "back_right", "back_left",]

clr_prompt = [f"a rendering image of 3D models, {view} view, color map." for view in VIEWS]
normal_prompt = [f"a rendering image of 3D models, {view} view, normal map." for view in VIEWS]
print ("max_model length", tokenizer.model_max_length)

for id, text_prompt in enumerate([clr_prompt, normal_prompt]):
    print(text_prompt)
    text_inputs = tokenizer(text_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(text_prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask,)
    prompt_embeds = prompt_embeds[0].detach().cpu()
    print(prompt_embeds.shape)

    embeds = {}
    for x, view in enumerate(VIEWS):
        embeds[view] = prompt_embeds[x]

    print(embeds)
    if id == 0:
        torch.save(embeds, f'./{path}/clr_embeds.pt')
    else:
        torch.save(embeds, f'./{path}/normal_embeds.pt')