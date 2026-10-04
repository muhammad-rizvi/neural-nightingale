# Neural Nightingale 🐦

**A bird species classifier that runs entirely in your browser.**
Take or upload a photo of a bird and a fine-tuned ResNet-18 names it as one of 15 North American species, with its top 3 guesses and how confident it is. Everything runs on your own device. The photo is never uploaded.

**Live demo:** <https://muhammad-rizvi.github.io/neural-nightingale/>

| | |
|---|---|
| Model | ResNet-18 (ImageNet-pretrained), fine-tuned |
| Classes | 15 bird species from CUB-200-2011 |
| Test accuracy | **96.9%** (217 / 224 held-out images) |
| Runs with | ONNX Runtime Web (WebAssembly), no server |
| Model size | ~45 MB |

---

## The 15 species

American Crow · American Goldfinch · Baltimore Oriole · Blue Jay · Brown Pelican · Cardinal · Common Yellowthroat · Downy Woodpecker · House Sparrow · Indigo Bunting · Mallard · Northern Flicker · Pileated Woodpecker · Red-winged Blackbird · Ruby-throated Hummingbird

The model can **only** answer with one of these 15. Show it a robin, a parrot, or a cat and it will still pick one of them. That's why the app shows a warning when its top confidence is below 80% (see [Confidence threshold](#confidence-threshold)).

---

## How it works

```
photo ──► resize shorter side to 256 ──► center-crop 224×224 ──► scale to 0–1
      ──► normalize (ImageNet mean/std) ──► ResNet-18 ──► 15 logits ──► softmax ──► top 3
```

1. **Training** happened in PyTorch on a free Colab GPU ([`neural_nightingale.ipynb`](neural_nightingale.ipynb)).
2. The trained model was **exported to ONNX** ([`scripts/export_onnx.py`](scripts/export_onnx.py)). ONNX is a standard file format that lots of runtimes can read, including one in the browser.
3. The web page uses **ONNX Runtime Web** to run `model.onnx` with WebAssembly, right on your device.
4. The page **preprocesses the photo the same way as in training** ([`preprocess.js`](preprocess.js)). This step is easy to get wrong. If the browser resized or normalized photos even a little differently from training, the model would quietly see slightly different inputs and accuracy would drop.

### Preprocessing parity

`preprocess.js` is a line-by-line port of the evaluation transform used in the notebook:

```python
eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

It doesn't rely on the browser's built-in canvas resizing, which uses a different algorithm. It copies **Pillow's antialiased bilinear filter** with the same fixed-point integer math, uses **Python's round-half-to-even** for the crop offsets, and does **float32 math** for normalization.

It was checked against torchvision on 20 test images (PNG and JPEG; sizes from 256×256 up to 4032×3024, including odd sizes, portrait, landscape, and images that need upscaling). In Chrome, the browser-made tensors matched PyTorch's **exactly**, with 0 of 150,528 values different on every image. The final logits matched Python ONNX Runtime to about 1e-6.

> Two small, deliberate differences: the browser rotates phone photos upright using their EXIF orientation (Pillow's `Image.open` doesn't), and images over 16 megapixels are scaled down first to fit within mobile Safari's canvas limit. Other browsers' JPEG decoders (e.g. Safari) may also round a few pixel values differently from Pillow.

---

## Training process

### Data
- **Dataset:** [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/): 11,788 photos of 200 bird species.
- **Subset:** 15 common, visually distinct North American species.
- **Splits:** the dataset's official *train* images were used for training. The official *test* images were split 50/50 at random (seed 42) into validation and test sets:

| Split | Images |
|---|---|
| Train | 450 |
| Validation | 223 |
| Test | 224 |

### Augmentation (training only)
`RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`, `ColorJitter(0.2, 0.2, 0.2)`, then ImageNet normalization. Validation and test images use the deterministic `eval_tfms` shown above.

### Two-stage transfer learning

**Stage 1: train only the new head.** All ResNet-18 weights were frozen and the final 1000-class layer was replaced with a new 512 → 15 linear layer (7,695 trainable parameters). Adam, lr = 1e-3, batch 32, 8 epochs.

| Epoch | Train acc | Val acc |
|---|---|---|
| 1 | 21.8% | 46.2% |
| 4 | 83.8% | 87.4% |
| 7 | 91.6% | **91.5%** ← best |
| 8 | 91.3% | 90.1% |

**Stage 2: fine-tune `layer4`.** Starting from the best stage-1 weights, the last residual stage was unfrozen (8.4M trainable parameters). It used differential learning rates: 1e-4 for `layer4` (small steps, so the pretrained features aren't wrecked) and 1e-3 for the head. 8 epochs.

| Epoch | Train acc | Val acc | Val loss |
|---|---|---|---|
| 1 | 91.6% | 93.3% | 0.290 |
| 3 | 99.8% | **95.5%** ← best | 0.183 |
| 8 | 99.3% | 93.3% | 0.218 |

Training accuracy hit ~99–100% while validation peaked at epoch 3, a sign that the model started to overfit later. The epoch-3 checkpoint was kept.

---

## Results

**Test accuracy: 96.9%** (217 of 224, test loss 0.154). The test set was used only once, at the very end.

With only 224 test images, that number has a standard error of about ±1.2 percentage points (√(p(1−p)/n)), so read it as "roughly 95–98%".

| Species | Precision | Recall | F1 |
|---|---|---|---|
| American Crow | 0.938 | 1.000 | 0.968 |
| American Goldfinch | 0.875 | 0.933 | 0.903 |
| Baltimore Oriole | 1.000 | 0.867 | 0.929 |
| Blue Jay | 1.000 | 1.000 | 1.000 |
| Brown Pelican | 1.000 | 1.000 | 1.000 |
| Cardinal | 1.000 | 1.000 | 1.000 |
| Common Yellowthroat | 0.938 | 1.000 | 0.968 |
| Downy Woodpecker | 1.000 | 1.000 | 1.000 |
| House Sparrow | 1.000 | 0.933 | 0.966 |
| Indigo Bunting | 1.000 | 1.000 | 1.000 |
| Mallard | 0.833 | 1.000 | 0.909 |
| Northern Flicker | 1.000 | 0.933 | 0.966 |
| Pileated Woodpecker | 1.000 | 1.000 | 1.000 |
| Red-winged Blackbird | 1.000 | 0.867 | 0.929 |
| Ruby-throated Hummingbird | 1.000 | 1.000 | 1.000 |
| **Macro average** | **0.972** | **0.969** | **0.969** |

The 7 mistakes: 2 Baltimore Orioles, 2 Red-winged Blackbirds, and 1 each of American Goldfinch, House Sparrow, and Northern Flicker were labelled as something else. The most common wrong answer was Mallard (3 times), then American Goldfinch (2).

### Confidence threshold

On the validation set, the model's confidence looked like this:

- **Correct predictions:** median confidence 97.3%. 10% of them were below 71.3%, and 25% were below 88.9%.
- **Wrong predictions:** 9 of the 10 had confidence under 60% (the other was 85.8%).

So the app shows *"Not confident — this may not be one of the 15 species I know"* whenever the top probability is below **0.80**. That flags almost every mistake, at the cost of also flagging some correct but uncertain answers. It matters even more for photos of birds outside the 15 species, where any answer is wrong.

---

## Project structure

| File | What it does |
|---|---|
| `index.html` | Page structure: photo "portal", results, species list, credits |
| `style.css` | All visual styling (dark mossy theme, mobile-first layout) |
| `app.js` | Loads the model, handles photos, runs inference, softmax, shows the top 3 |
| `preprocess.js` | Exact JavaScript port of `eval_tfms` |
| `model.onnx` | The trained network, exported from PyTorch |
| `class_names.json` | The 15 labels, in the same order as the model's outputs |
| `icon.svg` | Logo / favicon |
| `.nojekyll` | Tells GitHub Pages to serve files as-is |
| `scripts/export_onnx.py` | Rebuilds `model.onnx` + `class_names.json` from `model.pth` |
| `neural_nightingale.ipynb` | The full training notebook |
| `app.py`, `requirements.txt` | An earlier Gradio version of the demo (for Hugging Face Spaces) |

## Run it locally

Browsers won't load the model from a page opened by double-clicking (a `file://` page), so start a tiny local server in the project folder:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Credits & license

- **Dataset:** C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie. *The Caltech-UCSD Birds-200-2011 Dataset.* Computation & Neural Systems Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. Used here for educational, non-commercial purposes. The images belong to their original photographers.
- **Model:** ResNet-18 ([He et al., 2016](https://arxiv.org/abs/1512.03385)) with ImageNet weights from torchvision.
- **Runtime:** [ONNX Runtime Web](https://onnxruntime.ai/), MIT License.
- **Fonts:** Instrument Serif and DM Mono, via Google Fonts (SIL Open Font License).

This is a learning project. The predictions are for fun and education and shouldn't be relied on for serious bird identification.
