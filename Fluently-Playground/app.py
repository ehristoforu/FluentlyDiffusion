#!/usr/bin/env python

import os
import random
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

DESCRIPTION = """
# [**Fluently** Playground](https://huggingface.co/fluently)

"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max

USE_TORCH_COMPILE = 0
ENABLE_CPU_OFFLOAD = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    pipe_3_5 = StableDiffusionPipeline.from_pretrained(
        "fluently/Fluently-v3.5",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe_3_5.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_3_5.scheduler.config)
    pipe_3_5.to(device)   

    pipe_anime = StableDiffusionPipeline.from_pretrained(
        "fluently/Fluently-anime",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe_anime.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_anime.scheduler.config)
    pipe_anime.to(device)
    
    pipe_epic = StableDiffusionPipeline.from_pretrained(
        "fluently/Fluently-epic",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe_epic.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_epic.scheduler.config)
    pipe_epic.to(device)

    pipe_xl = StableDiffusionXLPipeline.from_pretrained(
        "fluently/Fluently-XL-v2",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe_xl.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_xl.scheduler.config)
    pipe_xl.to(device)

    print("Loaded on Device!")



def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(
    model,
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):

    
    seed = int(randomize_seed_fn(seed, randomize_seed))

    if not use_negative_prompt:
        negative_prompt = ""  # type: ignore
    if model == "Fluently v3.5":
        images = pipe_3_5(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
            num_images_per_prompt=1,
            output_type="pil",
        ).images
    elif model == "Fluently Anime":
        images = pipe_anime(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
            num_images_per_prompt=1,
            output_type="pil",
        ).images
    elif model == "Fluently Epic":
        images = pipe_epic(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
            num_images_per_prompt=1,
            output_type="pil",
        ).images
    else:
        images = pipe_xl(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=25,
            num_images_per_prompt=1,
            output_type="pil",
        ).images

    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed

        


examples = [
    "neon holography crystal cat",
    "a cat eating a piece of cheese",
    "an astronaut riding a horse in space",
    "a cartoon of a boy playing with a tiger",
    "a cute robot artist painting on an easel, concept art",
    "a close up of a woman wearing a transparent, prismatic, elaborate nemeses headdress, over the should pose, brown skin-tone"
]

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''
with gr.Blocks(title="Fluently Playground", css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=False,
    )

    with gr.Row():
        model = gr.Radio(
            label="Model",
            choices=["Fluently XL v2", "Fluently v3.5", "Fluently Anime", "Fluently Epic"],
            value="Fluently v3.5",
            interactive=True,
        )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, preview=True, show_label=False)
    with gr.Accordion("Advanced options", open=False):
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=5,
            value="""(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation""",
            placeholder="Enter a negative prompt",
            visible=False,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            visible=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=1536,
                step=8,
                value=768,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=1536,
                step=8,
                value=768,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                value=5.5,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=False,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )
    

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            model,
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    
if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_api=False, debug=False, inbrowser=True)