// preprocess.js — a JavaScript copy of the notebook's `eval_tfms`:
//
//   transforms.Resize(256)          -> resizeShorterSide()
//   transforms.CenterCrop(224)      -> centerCrop()
//   transforms.ToTensor()           -> toNormalizedCHW()  (0–255 -> 0–1, HWC -> CHW)
//   transforms.Normalize(mean, std) -> toNormalizedCHW()
//
// "Close enough" isn't the goal here. The model learned from tensors made by
// torchvision + Pillow, so this file repeats Pillow's resize arithmetic step by
// step (same filter, same rounding, same integer math). For the same decoded
// pixels it gives the same float32 tensor that PyTorch would build.

export const RESIZE_TO = 256;
export const CROP_SIZE = 224;
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];

// ---------------------------------------------------------------------------
// Step 1 — Resize(256): make the SHORTER side 256 px and keep the aspect ratio.
// ---------------------------------------------------------------------------

// Same formula as torchvision's _compute_resized_output_size.
// Python's int() drops the decimals, and so does Math.trunc().
export function resizedSize(width, height, shortSide = RESIZE_TO) {
  if (width <= height) return [shortSide, Math.trunc((shortSide * height) / width)];
  return [Math.trunc((shortSide * width) / height), shortSide];
}

// Pillow's resize does fixed-point integer math: every filter weight is scaled
// by 2^22 so the sums can stay whole numbers (see Pillow's src/libImaging/Resample.c).
const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;

// The "bilinear" filter is a triangle: weight 1 at the center, 0 at distance 1.
function bilinearFilter(x) {
  x = Math.abs(x);
  return x < 1 ? 1 - x : 0;
}

// For each output pixel along one axis: which input pixels it reads (bounds)
// and how much each one counts (weights). When we shrink an image, Pillow widens
// the triangle so every input pixel adds to the result (this is "antialiasing").
// Without that, shrunk photos turn grainy.
function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 1 * filterScale; // bilinear support is 1, scaled up when shrinking
  const kSize = Math.ceil(support) * 2 + 1;

  const bounds = new Int32Array(outSize * 2);
  const weights = new Int32Array(outSize * kSize);
  const tmp = new Float64Array(kSize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin; // from here on, xmax is "how many input pixels"

    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const w = bilinearFilter((x + xmin - center + 0.5) / filterScale);
      tmp[x] = w;
      total += w;
    }
    for (let x = 0; x < xmax; x++) {
      const w = total !== 0 ? tmp[x] / total : tmp[x]; // weights sum to 1
      // Convert to fixed point, rounding half away from zero (as Pillow does)
      weights[xx * kSize + x] = w < 0 ? Math.trunc(-0.5 + w * ONE) : Math.trunc(0.5 + w * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, weights, kSize };
}

// Convert a fixed-point sum back to a 0–255 byte (Pillow's clip8).
function clip8(v) {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return v >> PRECISION_BITS;
}

// Pillow resizes in two passes: first horizontally, then vertically, saving
// whole 0–255 bytes in between. We do the same so the rounding matches.
function resampleHorizontal(src, srcW, h, dstW) {
  const { bounds, weights, kSize } = precomputeCoeffs(srcW, dstW);
  const out = new Uint8ClampedArray(dstW * h * 4);
  const half = ONE / 2; // starting at 0.5 makes the final >> round to nearest
  for (let y = 0; y < h; y++) {
    const row = y * srcW * 4;
    for (let xx = 0; xx < dstW; xx++) {
      const xmin = bounds[xx * 2], n = bounds[xx * 2 + 1], k = xx * kSize;
      let r = half, g = half, b = half;
      for (let x = 0; x < n; x++) {
        const w = weights[k + x], p = row + (xmin + x) * 4;
        r += src[p] * w;
        g += src[p + 1] * w;
        b += src[p + 2] * w;
      }
      const o = (y * dstW + xx) * 4;
      out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b); out[o + 3] = 255;
    }
  }
  return out;
}

function resampleVertical(src, w, srcH, dstH) {
  const { bounds, weights, kSize } = precomputeCoeffs(srcH, dstH);
  const out = new Uint8ClampedArray(w * dstH * 4);
  const half = ONE / 2;
  for (let yy = 0; yy < dstH; yy++) {
    const ymin = bounds[yy * 2], n = bounds[yy * 2 + 1], k = yy * kSize;
    for (let x = 0; x < w; x++) {
      let r = half, g = half, b = half;
      for (let y = 0; y < n; y++) {
        const wt = weights[k + y], p = ((ymin + y) * w + x) * 4;
        r += src[p] * wt;
        g += src[p + 1] * wt;
        b += src[p + 2] * wt;
      }
      const o = (yy * w + x) * 4;
      out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b); out[o + 3] = 255;
    }
  }
  return out;
}

// Input and output are "RGBA images": { data: Uint8ClampedArray, width, height },
// the same shape as a canvas ImageData. Alpha is ignored, like .convert("RGB").
export function resizeShorterSide(img, shortSide = RESIZE_TO) {
  const [outW, outH] = resizedSize(img.width, img.height, shortSide);
  let data = img.data;
  if (outW !== img.width) data = resampleHorizontal(data, img.width, img.height, outW);
  if (outH !== img.height) data = resampleVertical(data, outW, img.height, outH);
  return { data, width: outW, height: outH };
}

// ---------------------------------------------------------------------------
// Step 2 — CenterCrop(224): cut the 224×224 square out of the middle.
// ---------------------------------------------------------------------------

// torchvision uses Python's round(), which rounds exact halves to the nearest
// EVEN number (round(2.5) == 2, round(3.5) == 4). JavaScript's Math.round(2.5)
// gives 3, so we write our own. A 1-pixel shift would change the tensor.
export function pythonRound(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function centerCrop(img, size = CROP_SIZE) {
  const top = pythonRound((img.height - size) / 2);
  const left = pythonRound((img.width - size) / 2);
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    const start = ((top + y) * img.width + left) * 4;
    out.set(img.data.subarray(start, start + size * 4), y * size * 4);
  }
  return { data: out, width: size, height: size };
}

// ---------------------------------------------------------------------------
// Steps 3 + 4 — ToTensor() and Normalize(): bytes -> normalized float32, CHW.
// ---------------------------------------------------------------------------

// The canvas gives pixels as R,G,B,A,R,G,B,A,... (HWC, "channels last").
// PyTorch wants every red value first, then all green, then all blue (CHW).
// For each value:  x = pixel / 255;  x = (x - mean[c]) / std[c]
// Math.fround rounds to float32 after each step, matching PyTorch's float32 math.
export function toNormalizedCHW(img) {
  const plane = img.width * img.height;
  const out = new Float32Array(3 * plane);
  const f = Math.fround;
  for (let c = 0; c < 3; c++) {
    const m = f(MEAN[c]), s = f(STD[c]);
    for (let i = 0; i < plane; i++) {
      const x = f(img.data[i * 4 + c] / 255);
      out[c * plane + i] = f(f(x - m) / s);
    }
  }
  return out;
}

// The whole eval_tfms pipeline. Returns the tensor data for the model
// (length 1*3*224*224) plus the 224×224 crop, which the page shows as a preview.
export function preprocess(img) {
  const resized = resizeShorterSide(img);
  const cropped = centerCrop(resized);
  return { tensor: toNormalizedCHW(cropped), crop: cropped };
}
