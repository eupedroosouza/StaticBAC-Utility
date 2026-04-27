<h1 align="center">StaticBAC Utility</h1>
<p align="center">A utility tool to execute StaticBAC tests</p>

See more of StaticBAC on your [repository](https://github.com/Jiovana/StaticBAC).

## Setup

### Prerequisites

* Python (version >= 3.14)

### Installing dependencies

1. **(If you have a NVIDIA GPU)** Recommended install torchvision with CUDA support:
    1. Check your CUDA maximum supported version:
    ```bash
    nvidia-smi
    ```
    2. Install torchvision based in your supported CUDA version:
   ```shell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu{your maximum supported cuda version}
   ```
2. Install python requirements (make sure if you are in StaticBAC-Utility folder):
   ```shell
   pip install -r requirements.txt
   ```

After that steps you are ready to use **StaticBAC-Utility**.

## Configuration
You can configure all the tool in config.yml file.
1. You can change output folder of results.
2. You need to modify (if necessary) the StaticBAC folder. You also need to have StaticBAC compiled in that folder (or specific executable path).
3. You can add (or remove, change, etc...) others models.
4. You need set the ImageNet Validation Dataset (links for Kaggle: [link1](https://www.kaggle.com/datasets/tusonggao/imagenet-validation-dataset), [link2](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview)) folder to inference in visual models. (if you need)

**config.yml**:
```yaml
# StaticBAC Utility Tool Configuration

utility:
  output_dir: ./Output


static_bac:
  folder: ../StaticBAC
  # If it is null, the algorithm search a executable on static-bac folder
  executable: null

models:
  gpt2:
    name: gpt2
    type: hf
    quantized: false
    inference: MediaWiki
  roBERTa:
    name: roberta-base
    type: hf
    quantized: false
    inference: sts-2
  google-t5:
    name: t5-base
    type: hf
    quantized: false
    inference: null
  inception-v3:
    name: inception_v3
    type: torchvision
    quantized: true
    weights: Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1
    inference: ImageNet
  mobilenet-v3:
    name: mobilenet_v3_large
    type: torchvision
    quantized: true
    weights: MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
    inference: ImageNet
  vgg-19:
    name: vgg19
    type: torchvision
    quantized: false
    weights: VGG19_Weights.IMAGENET1K_V1
    inference: ImageNet
  swin-v2-b:
    name: swin_v2_b
    type: torchvision
    quantized: false
    weights: Swin_V2_B_Weights.IMAGENET1K_V1
    inference: ImageNet
  # Already tested
  bert:
    name: bert-base-uncased
    type: hf
    quantized: false
    inference: null
  gpt:
    name: openai-gpt
    type: hf
    quantized: false
    inference: MediaWiki
  resnet50:
    name: resnet50
    type: torchvision
    quantized: false
    weights: ResNet50_Weights.IMAGENET1K_V2
    inference: ImageNet
  efficientnet-b0:
    name: efficientnet_b0
    type: torchvision
    quantized: false
    weights: EfficientNet_B0_Weights.IMAGENET1K_V1
    inference: ImageNet
  vit-b-16:
    name: vit_b_16
    type: torchvision
    quantized: false
    weights: ViT_B_16_Weights.IMAGENET1K_V1
    inference: ImageNet

dataset:
  image_net: ../Datasets/imagenet_validation/
```

## Usage
1. Start tool with (in tool folder):
   ```shell
   python .
   ```
2. Select the model what you want to do tests.
3. Place as many times you want to run the StaticBAC Encode/Decode phase.

After this, the model will be loaded and saved to encode, the encode/decode will be started, the model will be reconstructed, for last, the inference to obtain metric of encoded/decoded model will be run.\
The results are saved in output folder as `Output/results/{model}/{timestamp}/`:
* `reconstruction_errors.csv`: A list with reconstructions errors.
* `results.csv`: The results of encode/decode (some fields with average of executions) and inference metric.
* Other files are model files such as decoded model, reconstructed model and more.

## Observations
1. If you change the meta script (e.g: change the bitwidth of tensors), you need delete the model cached folder on: `Output/models/{model name}`. (the models are saved and checked with checksum to prevents load/download model always)
