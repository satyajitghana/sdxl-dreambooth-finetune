# SDXL - LoRA - DreamBooth

## 🧪 Development

```
black .
isort .
```

```
pip install -r requirements.txt
```

Tested with

```
torch==2.2.0.dev20231201+cu121
transformers==4.35.2
https://github.com/huggingface/diffusers.git@6bf1ca2c799f3f973251854ea3c379a26f216f36
typer==0.9.0
accelerate==0.24.1
rich==12.5.1
compel==2.0.2
```

## 🔥 Fine Tune

```
accelerate launch main.py dreambooth --input-images-dir ./data/tresa-truck --instance-prompt "a photo of a ohwx truck" --resolution 512 --train-batch-size 1 --max-train-steps 1000 --mixed-precision fp16 --output-dir ./output/tresa-truck
```

## 🍺 Infer

```
python main.py infer --prompt "a photo of a ohwx truck in a jungle" --lora-weights ./output/tresa-truck --output-dir output/infer-truck
```

## 😁 Outputs

![pharaoh](assets/pharaoh.png)

![pharaoh-1](assets/pharaoh-1.png)

![white-suit](assets/white-suite.JPG)

![marvel](assets/marvel.JPG)

![marvel](assets/marvel-1.png)

![dino](assets/dino.JPG)

![astro](assets/astro.JPG)


## Stable Diffusion Video Output

SDXL Generated Image

![dog](assets/sdv/dog.jpeg)

SD Video

![dog-video](assets/sdv/dog-sdv.gif)
