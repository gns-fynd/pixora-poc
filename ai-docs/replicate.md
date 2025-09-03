Here’s a developer-friendly, copy-pasteable **integration guide** for both Replicate models you linked—**Kling v2.0** (text/image → video) and **Nano Banana** (Gemini 2.5 Flash Image; image gen + editing). It covers model inputs/outputs, code for Python/Node/cURL, version pinning, files, sync vs async, and production tips.

---

# Replicate Models Integration Guide

## Models covered

* **kwaivgi/kling-v2.0** — text/image → short video (5s/10s, 720p). ([Replicate][1])
* **google/nano-banana** — Gemini 2.5 Flash Image, Google’s latest image generation & editing model. ([Replicate][2])

---

## 0) Auth & Clients (Replicate API)

* Get a **REPLICATE\_API\_TOKEN**, then use the Python/Node clients or raw HTTP. ([Replicate][3])
* **Sync vs polling**: add header `Prefer: wait=60` to block until completion (up to 60s), else poll with `predictions.get`. ([Replicate][4], [sdks.replicate.com][5])
* **Files**: pass via HTTP URLs or **data URIs**; the Node client can upload local files (Blob/File/Buffer). ([Replicate][6])
* **Python client tip**: `replicate.run()` returns a `FileOutput` for file results since v1.0; set `use_file_output=False` to get URLs. ([PyPI][7])

---

## 1) Model: **Kling v2.0** (kwaivgi/kling-v2.0)

### What it does

Generate **5s/10s** videos (720p) from a text prompt or from a **start image** (image-to-video). ([Replicate][1])

### Inputs (version: `03c47b84…` as example)

From the versioned API page (recommended for determinism):

* `prompt: string` — text prompt for video generation
* `negative_prompt: string` — things to avoid
* `aspect_ratio: "16:9" | ...` — **default 16:9**; **ignored if `start_image` is provided**
* `start_image: string` — URL or data URI for first frame (optional)
* `cfg_scale: number` — **default 0.5**, **max 1**; higher = stronger adherence, lower flexibility
* `duration: number` — **default 5** (seconds)
  (All straight from the model’s **Input schema** page.) ([Replicate][8])

**Output**: a **URI string** to the generated video file. ([Replicate][8])

### Python (pin version)

```python
import replicate, os
os.environ["REPLICATE_API_TOKEN"] = "<YOUR_TOKEN>"

MODEL = "kwaivgi/kling-v2.0:03c47b845aed8a009e0f83a45be0a2100ca11a7077e667a33224a54e85b2965c"  # pin version
out = replicate.run(
    MODEL,
    input={
        "prompt": "Aerial city, neon-lit skyscrapers, flying cars, camera follows subject",
        "duration": 5,
        "cfg_scale": 0.5,
        "aspect_ratio": "16:9",
        "negative_prompt": ""
    },
    # use_file_output=False → return URL instead of FileOutput (optional)
)
print(out)  # video URL or FileOutput
```

(General Python usage + file-handling behavior documented in Replicate’s Python guide and PyPI notes.) ([Replicate][3], [PyPI][7])

### Node.js

```ts
import Replicate from "replicate";
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

const out = await replicate.run(
  "kwaivgi/kling-v2.0:03c47b845aed8a009e0f83a45be0a2100ca11a7077e667a33224a54e85b2965c",
  {
    input: {
      prompt: "Macro shot of espresso pour, slow orbit, cinematic lighting",
      duration: 5,
      cfg_scale: 0.5,
      aspect_ratio: "16:9"
    }
  }
);
console.log(out); // URL
```

**Local file as `start_image`** (Node): pass a Blob/File/Buffer—SDK handles upload; or use a **data URI** for small files. ([Replicate][9])

### cURL (sync mode)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait=60" \
  -d '{
    "input": {
      "prompt": "Studio product spin of a smartwatch, soft rim light",
      "duration": 5,
      "cfg_scale": 0.5,
      "aspect_ratio": "16:9"
    }
  }' \
  https://api.replicate.com/v1/models/kwaivgi/kling-v2.0/predictions
```

(Prefer header behavior & predictions endpoints per Replicate docs.) ([Replicate][10])

### Notes & policies

* **Privacy/terms/SLA** for Kling are hosted by Kuaishou; Replicate forwards data accordingly (see README links). ([Replicate][1])

---

## 2) Model: **Nano Banana** (google/nano-banana)

### What it does

Google’s **Gemini 2.5 Flash Image** model for **native image generation + precise image editing** (multi-image fusion, character/style consistency, conversational edits). **SynthID watermarking** is applied to generated/edited images. ([Replicate][2])

### Inputs (version: `f0a9d34b…` as example)

From the versioned API page:

* `prompt: string` — text description of the image to generate
* `image_input: array` — **one or more input images** to **transform** or use as **reference**
* `output_format: "jpg" | "png" | ...` — **default `jpg`**
  (Fields listed on the model’s **Input schema** page; additional fields may appear on newer versions.) ([Replicate][11])

**Output**: a **URI string** to the generated/edited image. ([Replicate][11])

### Python (text-to-image)

```python
import replicate, os
os.environ["REPLICATE_API_TOKEN"] = "<YOUR_TOKEN>"

MODEL = "google/nano-banana:f0a9d34b12ad1c1cd76269a844b218ff4e64e128ddaba93e15891f47368958a0"
img_url = replicate.run(MODEL, input={
    "prompt": "Editorial product shot of sneakers on concrete, soft shadow, 35mm look",
    "output_format": "jpg"
})
print(img_url)
```

(General Python steps from Replicate’s “Run a model from Python”.) ([Replicate][3])

### Python (image-to-image edit / multi-image fusion)

```python
img_url = replicate.run(MODEL, input={
    "prompt": "Replace background with a sunlit studio cyclorama, keep subject + lighting",
    "image_input": [
      "https://example.com/in/shoe1.jpg",
      "https://example.com/in/bg_texture.jpg"
    ],
    "output_format": "png"
})
```

(Editing + multi-image fusion supported by the model family; schema shows `image_input[]`.) ([Replicate][11])

### Node.js

```ts
import Replicate from "replicate";
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

const out = await replicate.run(
  "google/nano-banana:f0a9d34b12ad1c1cd76269a844b218ff4e64e128ddaba93e15891f47368958a0",
  {
    input: {
      prompt: "Portrait photo, cinematic key light, shallow depth of field",
      output_format: "jpg"
    }
  }
);
console.log(out); // image URL
```

### cURL (edit an existing image)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait=60" \
  -d '{
    "input": {
      "prompt": "Remove background, keep soft shadows; place on off-white studio paper",
      "image_input": ["https://example.com/in/product.png"],
      "output_format": "png"
    }
  }' \
  https://api.replicate.com/v1/models/google/nano-banana/predictions
```

(Prefer header & prediction flow per Replicate HTTP docs.) ([Replicate][10], [sdks.replicate.com][5])

### Notes

* **Watermarking**: Google’s SynthID watermark is embedded for provenance. ([Replicate][2])

---

## 3) Version pinning & schema discovery

* Prefer **pinning a specific version hash** (`owner/model:VERSION`) for reproducibility. Use the model page → **Versions** → copy hash → open `/versions/<hash>/api` for the exact **Input schema** and **Output schema**. (Examples shown above for both models.) ([Replicate][8])
* If using latest, the generic “API” tab shows the SDK snippets; but **inputs can change** across versions—pin to avoid breakage. ([Replicate][12])

---

## 4) Files: URLs, data URIs, or SDK uploads

* **HTTP URLs** for big files; **data URIs** fine for small (<\~256KB). Node SDK can upload local files via Blob/File/Buffer. ([Replicate][6])

---

## 5) Handling long jobs & webhooks

* For tasks that exceed the `Prefer: wait` window, poll `predictions.get`.
* You can also provide a **webhook** and filter events (`start`, `output`, `logs`, `completed`) to drive UI updates. ([sdks.replicate.com][5], [Replicate][13])

---

## 6) Production tips

* **Determinism**: pin model version; for Kling, control adherence with `cfg_scale` (≤1). ([Replicate][8])
* **Parallelism**: fan-out multiple predictions client-side (e.g., per scene) and gather results.
* **Artifacts**: Replicate purges inputs/outputs/logs after a period—**persist your download URLs or files** if you need them. ([api.replicate.com][14])
* **Pricing & limits**: check each model’s **Pricing** tab on Replicate for current rates/quotas. (Model pages link it beside API/Playground.) ([Replicate][12])

---

## 7) Drop-in “tool” wrappers (for your agent)

### Kling tool (text → video or image → video)

```python
from typing import Optional
import replicate

KLING = "kwaivgi/kling-v2.0:03c47b845aed8a009e0f83a45be0a2100ca11a7077e667a33224a54e85b2965c"

def kling_video(prompt: str,
                duration: int = 5,
                aspect_ratio: Optional[str] = "16:9",
                cfg_scale: float = 0.5,
                negative_prompt: str = "",
                start_image: Optional[str] = None) -> str:
    """
    Return URL (or FileOutput) to generated video.
    """
    inp = {
        "prompt": prompt,
        "duration": duration,
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt
    }
    if aspect_ratio and not start_image:
        inp["aspect_ratio"] = aspect_ratio
    if start_image:
        inp["start_image"] = start_image
    return replicate.run(KLING, input=inp)
```

(Parameters mirror the model’s Input schema; `aspect_ratio` ignored if `start_image` is set.) ([Replicate][8])

### Nano Banana tool (image gen / edit / fusion)

```python
NANO = "google/nano-banana:f0a9d34b12ad1c1cd76269a844b218ff4e64e128ddaba93e15891f47368958a0"

def nano_image(prompt: str,
               image_input: list[str] | None = None,
               output_format: str = "jpg") -> str:
    """
    Return URL (or FileOutput) to generated/edited image.
    If image_input is provided, performs edit/fusion; else pure text-to-image.
    """
    inp = {"prompt": prompt, "output_format": output_format}
    if image_input:
        inp["image_input"] = image_input
    return replicate.run(NANO, input=inp)
```

(Fields per version Input schema; output is a single URI.) ([Replicate][11])

---

## 8) Quick smoke tests

```bash
# Kling — quick 5s clip
python - <<'PY'
import os, replicate
os.environ["REPLICATE_API_TOKEN"] = "<TOKEN>"
m = "kwaivgi/kling-v2.0:03c47b845aed8a009e0f83a45be0a2100ca11a7077e667a33224a54e85b2965c"
print(replicate.run(m, input={"prompt":"Golden retriever running on the beach, camera follows", "duration":5, "cfg_scale":0.5, "aspect_ratio":"16:9"}))
PY

# Nano Banana — edit with one reference image
python - <<'PY'
import os, replicate
os.environ["REPLICATE_API_TOKEN"] = "<TOKEN>"
m = "google/nano-banana:f0a9d34b12ad1c1cd76269a844b218ff4e64e128ddaba93e15891f47368958a0"
print(replicate.run(m, input={"prompt":"Replace background with clean studio paper, keep soft shadow", "image_input":["https://example.com/in.png"], "output_format":"png"}))
PY
```

---

### References

* Kling v2.0 model page/README (overview, policies). ([Replicate][1])
* Kling v2.0 **versioned API** (Input & Output schema). ([Replicate][8])
* Nano Banana model README (capabilities, watermark). ([Replicate][2])
* Nano Banana **versioned API** (Input & Output schema). ([Replicate][11])
* Replicate Python quickstart, HTTP API, predictions (Prefer: wait, polling), file handling & client behavior. ([Replicate][3], [sdks.replicate.com][5], [PyPI][7])

If you want, I can also turn these into **agent tools** (LangChain `@tool` functions) ready to drop into your ReACT graph.

[1]: https://replicate.com/kwaivgi/kling-v2.0/readme "
      
  
    AI Video | Readme and Docs
  

    "
[2]: https://replicate.com/google/nano-banana/readme "
      
  
    Nano Banana | Google | Image Editing | Readme and Docs
  

    "
[3]: https://replicate.com/docs/get-started/python?utm_source=chatgpt.com "Run a model from Python - Replicate docs"
[4]: https://replicate.com/docs/topics/predictions/create-a-prediction.md?utm_source=chatgpt.com "replicate.com"
[5]: https://sdks.replicate.com/resources/predictions/?utm_source=chatgpt.com "Documentation | Predictions - sdks.replicate.com"
[6]: https://replicate.com/kwaivgi/kling-v2.0/api/api-reference?utm_source=chatgpt.com "AI Video | API reference - replicate.com"
[7]: https://pypi.org/project/replicate/?utm_source=chatgpt.com "replicate · PyPI"
[8]: https://replicate.com/kwaivgi/kling-v2.0/versions/03c47b845aed8a009e0f83a45be0a2100ca11a7077e667a33224a54e85b2965c/api "
      
  kwaivgi/kling-v2.0:03c47b84 | Run with an API on Replicate

    "
[9]: https://replicate.com/kwaivgi/kling-v2.0/api/learn-more?utm_source=chatgpt.com "kwaivgi/kling-v2.0 – API reference - replicate.com"
[10]: https://replicate.com/docs/reference/http?utm_source=chatgpt.com "HTTP API - Replicate docs"
[11]: https://replicate.com/google/nano-banana/versions/f0a9d34b12ad1c1cd76269a844b218ff4e64e128ddaba93e15891f47368958a0/api "
      
  google/nano-banana:f0a9d34b | Run with an API on Replicate

    "
[12]: https://replicate.com/kwaivgi/kling-v2.0/api "
      
  
    
      AI Video | API reference
    
  

    "
[13]: https://replicate.com/replicate/goo/api/api-reference?utm_source=chatgpt.com "replicate/goo | API reference"
[14]: https://api.replicate.com/openapi.json?utm_source=chatgpt.com "Replicate"
