import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

# ---------- 1. Rebuild the empty architecture, then load our trained weights ----------
checkpoint = torch.load("model.pth", map_location="cpu")
class_names = checkpoint["class_names"]
pretty = [c.replace("_", " ") for c in class_names]

model = models.resnet18(weights=None)                          # structure only, no download
model.fc = nn.Linear(model.fc.in_features, len(class_names))   # same 15-output head as training
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# ---------- 2. EXACTLY the same preprocessing as validation/test ----------
eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CONFIDENCE_THRESHOLD = 0.80

# ---------- 3. The prediction function Gradio calls on every upload ----------
def predict(img):
    if img is None:
        return {}, "Upload a photo to get started."
    x = eval_tfms(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    top = probs.topk(3)
    results = {pretty[i]: v.item() for v, i in zip(top.values, top.indices)}

    if top.values[0] < CONFIDENCE_THRESHOLD:
        note = ("**Not confident.** This may not be one of the 15 species I know, "
                "or the bird may be hard to see. Treat these guesses with caution.")
    else:
        note = f"Looks like a **{pretty[top.indices[0]]}**."
    return results, note

# ---------- 4. The web interface ----------
species_list = ", ".join(pretty)
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Bird photo"),
    outputs=[gr.Label(num_top_classes=3, label="Top 3 predictions"), gr.Markdown()],
    title="Neural Nightingale",
    description=(
        "A ResNet-18 fine-tuned on 15 North American bird species "
        "(~97% test accuracy). Upload a photo to see the top 3 guesses.\n\n"
        f"**Species it knows:** {species_list}.\n\n"
        "It can only choose among these 15, so any other bird (or a car!) "
        "will still get labeled as one of them. "
        "Trained on the CUB-200-2011 dataset (Caltech/UCSD), for educational use."
    ),
)

demo.launch()
