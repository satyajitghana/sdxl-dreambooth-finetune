from enum import Enum
from pathlib import Path

import typer
from rich import print
from typing_extensions import Annotated

from infer import infer_lora
from training.dreambooth import train as train_dreambooth

app = typer.Typer(pretty_exceptions_enable=False)


class MixedPrecisionType(str, Enum):
    no = "no"
    fp16 = "fp16"
    bf16 = "bf16"


@app.command(help="Fine Tune Stable Diffusion with LoRA and DreamBooth")
def dreambooth(
    input_images_dir: Annotated[
        str, typer.Option(help="Path to folder containing training data")
    ],
    instance_prompt: Annotated[
        str,
        typer.Option(
            help="The prompt with identifier specifying the instance, e.g. 'a photo of a ohwx man', 'a photo of a TOK man wearing casual clothes, smiling'"
        ),
    ],
    base_model: Annotated[
        str, typer.Option(help="Base Model to train Dreambooth on",),
    ] = "stabilityai/stable-diffusion-xl-base-1.0",
    pretrained_vae: Annotated[
        str, typer.Option(help="VAE model with better numerical stability",),
    ] = "madebyollin/sdxl-vae-fp16-fix",
    resolution: Annotated[
        int,
        typer.Option(
            help="The resolution for input images, all the images will be resized to this",
        ),
    ] = 1024,
    train_batch_size: Annotated[
        int, typer.Option(help="Batch Size (per device) for training")
    ] = 1,
    max_train_steps: Annotated[
        int,
        typer.Option(
            help="Total number of training steps to run for, more your images, more should be this value",
        ),
    ] = 500,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            help="Number of update steps to accumulate before performing a backward pass",
        ),
    ] = 1,
    learning_rate: Annotated[
        float,
        typer.Option(help="Initial learning rate for training, after warmup period"),
    ] = 1e-4,
    use_8bit_adam: Annotated[
        bool,
        typer.Option(
            help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
        ),
    ] = False,
    use_tf32: Annotated[
        bool,
        typer.Option(
            help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.",
        ),
    ] = False,
    mixed_precision: Annotated[
        MixedPrecisionType,
        typer.Option(
            help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.",
        ),
    ] = MixedPrecisionType.no,
    lora_rank: Annotated[
        int, typer.Option(help="The dimension of the LoRA update matrices")
    ] = 4,
    output_dir: Annotated[
        str,
        typer.Option(
            help="The output directory to store the logs, model predictions, checkpoints and final lora model weights",
        ),
    ] = "lora-dreambooth-model",
):
    train_dreambooth(
        input_images_dir,
        instance_prompt,
        base_model,
        pretrained_vae,
        resolution,
        train_batch_size,
        max_train_steps,
        gradient_accumulation_steps,
        learning_rate,
        use_8bit_adam,
        use_tf32,
        mixed_precision,
        lora_rank,
        output_dir,
    )


@app.command(help="Inference with Trained LoRA")
def infer(
    prompt: Annotated[
        str,
        typer.Option(
            help="The prompt for text to image, e.g. 'a photo of a ohwx man', 'a photo of a TOK man wearing casual clothes, smiling'"
        ),
    ],
    lora_weights: Annotated[
        str, typer.Option(help="Path to the lora safetensors, or the folder")
    ],
    base_model: Annotated[
        str, typer.Option(help="Base Model your DreamBooth was trained on",),
    ] = "stabilityai/stable-diffusion-xl-base-1.0",
    pretrained_vae: Annotated[
        str, typer.Option(help="VAE model with better numerical stability",),
    ] = "madebyollin/sdxl-vae-fp16-fix",
    output_dir: Annotated[
        str, typer.Option(help="The output directory to store the images",),
    ] = "infer-outputs",
):
    infer_lora(
        prompt, lora_weights, base_model, pretrained_vae, output_dir,
    )


if __name__ == "__main__":
    app()
