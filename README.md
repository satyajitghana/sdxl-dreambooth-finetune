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

```
07/12/2023 23:36:01 INFO     07/12/2023 23:36:01 - INFO - training.dreambooth - unet params = 2567463684                                                                         logging.py:60
07/12/2023 23:36:06 INFO     07/12/2023 23:36:06 - INFO - training.dreambooth - training lora parameters = 5806080                                                               logging.py:60
                    INFO     07/12/2023 23:36:06 - INFO - training.dreambooth - using torch AdamW                                                                                logging.py:60
                    INFO     07/12/2023 23:36:06 - INFO - training.dreambooth - 🚧 computing time ids                                                                            logging.py:60
                    INFO     07/12/2023 23:36:06 - INFO - training.dreambooth - precomputing text embeddings                                                                     logging.py:60
07/12/2023 23:36:07 INFO     07/12/2023 23:36:07 - INFO - training.dreambooth - 🏃 🏃 🏃 Training Config 🏃 🏃 🏃                                                                logging.py:60
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth -   Num examples = 8                                                                               logging.py:60
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth -   Num batches each epoch = 8                                                                     logging.py:60
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth -   Instantaneous batch size per device = 1                                                        logging.py:60
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth -   Total train batch size (w. parallel, distributed & accumulation) = 1                           logging.py:60
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth -   Gradient Accumulation steps = 1                                                                logging.py:60
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth -   Total optimization steps = 1000 
                    INFO     07/12/2023 23:36:07 - INFO - training.dreambooth - 🧪 start training... 
07/12/2023 23:47:19 INFO     07/12/2023 23:47:19 - INFO - training.dreambooth - ✅ training done!                                                                                logging.py:60
                    INFO     07/12/2023 23:47:19 - INFO - training.dreambooth - 🍉 saved lora weights in ./output/tresa-truck-2                                                  logging.py:60
                    INFO     07/12/2023 23:47:19 - INFO - training.dreambooth - 🎉 🎉 🎉 ALL DONE 🎉 🎉 🎉                                                                       logging.py:60
Steps 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:11:12 < 0:00:00 , 1 it/s ]
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
