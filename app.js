// app.js — wires the page together:
//   1. download model.onnx + class_names.json and start ONNX Runtime
//   2. when a photo arrives (button, camera, drag-and-drop, paste): decode it,
//      preprocess it (preprocess.js), run the model, apply softmax, show the top 3

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { preprocess, CROP_SIZE } from "./preprocess.js";

const MODEL_URL = "model.onnx";
const CLASSES_URL = "class_names.json";
const CONFIDENCE_THRESHOLD = 0.8; // same threshold as app.py
const MAX_PIXELS = 16_000_000;    // iPhone Safari can't read canvases bigger than ~16.7 MP

// Multi-threading needs special server headers that GitHub Pages can't send,
// so we tell ONNX Runtime up front to use one thread.
ort.env.wasm.numThreads = 1;

const $ = (id) => document.getElementById(id);
const portal = $("portal");
const photo = $("photo");
const statusEl = $("model-status");
const statusText = $("model-status-text");
const resultsEl = $("results");
const warningEl = $("warning");
const barsEl = $("bars");
const errorEl = $("error");
const speciesList = $("species-list");
const verdictName = $("verdict-name");
const verdictEyebrow = $("verdict-eyebrow");
const fileInput = $("file-input");
const cameraInput = $("camera-input");

let classNames = [];
let photoUrl = null;
let latestRun = 0; // makes sure only the newest photo's result is shown

// ---------------------------------------------------------------------------
// 1. Loading
// ---------------------------------------------------------------------------

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusText.textContent = text;
}

// fetch() that reports progress, so people on slow phones see something happening.
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url} (HTTP ${res.status})`);
  if (!res.body) return new Uint8Array(await res.arrayBuffer());

  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

const mb = (bytes) => (bytes / 1e6).toFixed(0);

async function loadClassNames() {
  const res = await fetch(CLASSES_URL);
  if (!res.ok) throw new Error(`Could not download ${CLASSES_URL} (HTTP ${res.status})`);
  classNames = await res.json();
  speciesList.innerHTML = "";
  classNames.forEach((name, i) => {
    const li = document.createElement("li");
    const num = document.createElement("span");
    num.textContent = String(i + 1).padStart(2, "0");
    li.append(num, name);
    speciesList.append(li);
  });
}

async function loadModel() {
  const bytes = await fetchWithProgress(MODEL_URL, (got, total) => {
    setStatus("loading", total ? `Loading model ${mb(got)} / ${mb(total)} MB` : `Loading model ${mb(got)} MB`);
  });
  setStatus("loading", "Starting model…");
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  setStatus("ready", "Model ready · on-device");
  return session;
}

const classNamesPromise = loadClassNames();
const sessionPromise = loadModel();

Promise.all([classNamesPromise, sessionPromise]).catch((err) => {
  console.error(err);
  setStatus("error", "Model failed to load");
  showError(
    location.protocol === "file:"
      ? "Browsers block loading the model from a file:// page. Start a local server in this folder (python3 -m http.server) and open http://localhost:8000."
      : `The model couldn't load: ${err.message}. Try refreshing the page.`
  );
});

// ---------------------------------------------------------------------------
// 2. Getting pixels out of a photo
// ---------------------------------------------------------------------------

// Turns an image file into raw RGBA pixels (a canvas ImageData).
//  - imageOrientation "from-image": phone photos come out upright (EXIF rotation)
//  - colorSpaceConversion "none": keep the file's pixel values as they are,
//    like Pillow, instead of letting the browser adjust colors for the screen
async function decodeImage(file) {
  let source;
  try {
    source = await createImageBitmap(file, {
      imageOrientation: "from-image",
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
  } catch {
    // Older Safari: fall back to a regular <img> element
    source = new Image();
    source.src = URL.createObjectURL(file);
    await source.decode();
    URL.revokeObjectURL(source.src);
  }

  let width = source.naturalWidth || source.width;
  let height = source.naturalHeight || source.height;
  // Very large photos (e.g. 48 MP) are first scaled to below the canvas limit.
  // Everything up to 16 MP, which covers most phone photos, is read untouched.
  if (width * height > MAX_PIXELS) {
    const scale = Math.sqrt(MAX_PIXELS / (width * height));
    width = Math.floor(width * scale);
    height = Math.floor(height * scale);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, width, height);
  source.close?.();
  return ctx.getImageData(0, 0, width, height);
}

// ---------------------------------------------------------------------------
// 3. Running the model
// ---------------------------------------------------------------------------

// Softmax turns raw scores (logits) into probabilities that add up to 1.
// Subtracting the max first doesn't change the answer but avoids overflow.
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

async function classify(file) {
  if (!file) return;
  const run = ++latestRun;
  hideError();
  showPhoto(file);
  portal.dataset.state = "busy";
  verdictEyebrow.textContent = "Analyzing";
  verdictName.textContent = "Looking closely…";

  try {
    // Let the browser paint the "busy" state before the heavy work starts
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

    const [session] = await Promise.all([sessionPromise, classNamesPromise]);
    const pixels = await decodeImage(file);
    const { tensor, crop } = preprocess(pixels);

    const input = new ort.Tensor("float32", tensor, [1, 3, CROP_SIZE, CROP_SIZE]);
    const output = await session.run({ image: input });
    const probs = softmax(output.logits.data);

    if (run !== latestRun) return; // a newer photo arrived meanwhile
    drawCrop(crop);
    showResults(probs);
  } catch (err) {
    if (run !== latestRun) return;
    console.error(err);
    portal.dataset.state = "idle";
    showError("Couldn't read that image. Try a JPEG or PNG photo.");
  }
}

// ---------------------------------------------------------------------------
// 4. Showing things on the page
// ---------------------------------------------------------------------------

const pct = (p) => `${(p * 100).toFixed(1)}%`;

function showPhoto(file) {
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  photo.src = photoUrl;
  photo.hidden = false;
  // Restart the fade-in animation for each new photo
  photo.style.animation = "none";
  void photo.offsetWidth;
  photo.style.animation = "";
}

function drawCrop(crop) {
  const ctx = $("crop-canvas").getContext("2d");
  ctx.putImageData(new ImageData(crop.data, crop.width, crop.height), 0, 0);
}

function showResults(probs) {
  // Indices of the 3 highest probabilities
  const top3 = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 3);
  const best = probs[top3[0]];
  const confident = best >= CONFIDENCE_THRESHOLD;

  portal.dataset.state = "result";
  verdictEyebrow.textContent = confident ? `Top match · ${pct(best)} confident` : `Best guess · only ${pct(best)}`;
  verdictName.textContent = classNames[top3[0]];

  warningEl.hidden = confident;
  resultsEl.classList.toggle("is-low", !confident);

  barsEl.innerHTML = "";
  for (const [rank, i] of top3.entries()) {
    const li = document.createElement("li");
    li.className = "bar-row";
    li.innerHTML = `
      <span class="bar-rank">0${rank + 1}</span>
      <span class="bar-name"></span>
      <span class="bar-pct">${pct(probs[i])}</span>
      <span class="bar-track"><span class="bar-fill"></span></span>`;
    li.querySelector(".bar-name").textContent = classNames[i];
    barsEl.append(li);
  }
  resultsEl.hidden = false;
  // Set widths one frame later so the bars animate from 0
  requestAnimationFrame(() => {
    barsEl.querySelectorAll(".bar-fill").forEach((bar, rank) => {
      bar.style.width = `${(probs[top3[rank]] * 100).toFixed(2)}%`;
    });
  });

  [...speciesList.children].forEach((li, i) => li.classList.toggle("is-match", confident && i === top3[0]));

  // On phones the results are below the photo, so scroll them into view
  if (window.matchMedia("(max-width: 899px)").matches) {
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}
function hideError() {
  errorEl.hidden = true;
}

// ---------------------------------------------------------------------------
// 5. Ways to give it a photo
// ---------------------------------------------------------------------------

$("choose-btn").addEventListener("click", () => fileInput.click());
$("camera-btn").addEventListener("click", () => cameraInput.click());

for (const input of [fileInput, cameraInput]) {
  input.addEventListener("change", () => {
    classify(input.files[0]);
    input.value = ""; // allows choosing the same file twice
  });
}

// Drag and drop onto the portal
portal.addEventListener("dragover", (e) => {
  e.preventDefault();
  portal.classList.add("is-dragging");
});
portal.addEventListener("dragleave", () => portal.classList.remove("is-dragging"));
portal.addEventListener("drop", (e) => {
  e.preventDefault();
  portal.classList.remove("is-dragging");
  classify(e.dataTransfer.files[0]);
});

// Paste an image from the clipboard (Cmd/Ctrl + V)
document.addEventListener("paste", (e) => {
  const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/"));
  if (file) classify(file);
});
