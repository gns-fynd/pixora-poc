# Firecrawl: extracting images from a URL (hosted & self-host)

This guide shows two reliable ways to pull image URLs (and related metadata) from a page with Firecrawl:

* **Single page**: `/v2/scrape` → return **raw HTML** (or do **one-pass JSON extraction**) and collect `<img>` data. ([docs.firecrawl.dev][1])
* **Multiple pages / whole site**: `/v2/extract` → ask Firecrawl’s LLM to return a **structured list of images** across pages, with polling for results. ([docs.firecrawl.dev][2])

It also covers dynamic pages (scroll/wait), avoiding base64 bloat, and how to **self-host** Firecrawl with your **own OpenAI key** for LLM-based extraction.

---

## When to use which

* **One page, full control:** `/v2/scrape`. Ask for `rawHtml` (or run a JSON format extraction in the same call) and parse image info. Supported formats include `markdown`, `links`, `html`, `rawHtml`, `summary`, plus object formats like `{ type: "json", ... }` and `{ type: "screenshot", ... }`. ([docs.firecrawl.dev][1])
* **Many pages / wildcard (`/*`)**: `/v2/extract` job with a **prompt + JSON schema**. You’ll get a job `id` then **GET** status to retrieve structured data. Supports `enableWebSearch`, wildcard URLs, and `ignoreInvalidURLs`. ([docs.firecrawl.dev][3])

> Rate limits & error codes are documented in the API intro. Expect 429 for rate limiting, 402 for payment required, and 5xx on server errors. ([docs.firecrawl.dev][4])

---

## Prereqs

* **Hosted**: Firecrawl API key (Authorization: `Bearer fc-...`).
* **Self-host**: run Firecrawl locally; set **`OPENAI_API_KEY`** in `.env` to enable LLM-based features (e.g., extraction). Firecrawl’s self-host guide shows required envs (Redis, Playwright service, etc.). Cloud-only “Fire-engine” isn’t available when self-hosting. ([docs.firecrawl.dev][5])

---

## Approach A — Single page via `/v2/scrape`

### A1) Scrape raw HTML and extract `<img>` tags yourself

Ask Firecrawl for `rawHtml` (and optionally `links` and a full-page `screenshot`). You can **focus content** with `includeTags` (e.g., `"img"`, gallery selectors) and exclude noise with `excludeTags`. Default cache is \~2 days via `maxAge`. By default, **`removeBase64Images: true`** to keep responses small. ([docs.firecrawl.dev][1])

**cURL**

```bash
curl --request POST https://api.firecrawl.dev/v2/scrape \
  --header "Authorization: Bearer $FIRECRAWL_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "url": "https://example.com/product/123",
    "formats": [
      "rawHtml",
      { "type": "screenshot", "fullPage": true, "quality": 80 }
    ],
    "includeTags": ["img", ".product-gallery", ".hero"],
    "excludeTags": ["#footer", ".ads"],
    "onlyMainContent": false,
    "removeBase64Images": true,
    "waitFor": 1500
  }'
```

Response includes `data.rawHtml`, `data.links`, and optional `data.screenshot`. Parse `rawHtml` for `<img src>`, `srcset`, `alt`. (See full response shape in docs.) ([docs.firecrawl.dev][6])

**Python (BeautifulSoup post-processing)**

```python
import requests, urllib.parse
from bs4 import BeautifulSoup

API = "https://api.firecrawl.dev/v2/scrape"
payload = {
  "url": "https://example.com/product/123",
  "formats": ["rawHtml"],
  "includeTags": ["img", ".product-gallery"],
  "onlyMainContent": False,
  "removeBase64Images": True,
  "waitFor": 1500
}
resp = requests.post(API, json=payload, headers={"Authorization": f"Bearer {YOUR_API_KEY}"})
resp.raise_for_status()
data = resp.json()["data"]

html = data["rawHtml"]
base = data["metadata"]["sourceURL"]  # helpful to resolve relative paths
soup = BeautifulSoup(html, "html.parser")

images = []
for img in soup.find_all("img"):
    src = img.get("src") or ""
    if not src or src.startswith("data:"):  # skip base64/data URIs (can toggle removeBase64Images=False if needed)
        continue
    abs_src = urllib.parse.urljoin(base, src)
    images.append({
        "src": abs_src,
        "alt": img.get("alt") or "",
        "srcset": img.get("srcset") or "",
        "sizes": img.get("sizes") or ""
    })

print({"images": images})
```

**Why this works well:** The `/scrape` endpoint explicitly returns `rawHtml` and `metadata.sourceURL`, and supports pre-scrape **actions** (`wait`, `scroll`, `click`, etc.) to load lazy images before capture. ([docs.firecrawl.dev][1])

> Tip for dynamic pages: add `actions` steps like `scroll` or `wait` before scraping, so lazy-loaded galleries render. ([docs.firecrawl.dev][1])

---

### A2) One-pass **JSON extraction** on `/v2/scrape` (no client-side parsing)

Firecrawl can return **structured JSON** directly by adding a JSON format object inside `formats`. Provide a **prompt + JSON schema** describing the image objects you want (e.g., `src`, `alt`, `is_primary`), and Firecrawl’s LLM will fill it. ([docs.firecrawl.dev][1])

**cURL**

```bash
curl -s -X POST https://api.firecrawl.dev/v2/scrape \
  -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/product/123",
    "formats": [{
      "type": "json",
      "prompt": "Extract all product image URLs on the page with alt texts. Resolve relative paths.",
      "schema": {
        "type": "object",
        "properties": {
          "images": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "src": {"type": "string"},
                "alt": {"type": "string"},
                "is_primary": {"type": "boolean"}
              },
              "required": ["src"]
            }
          }
        },
        "required": ["images"]
      }
    }],
    "onlyMainContent": false,
    "removeBase64Images": true
  }'
```

That returns a JSON blob (in the `formats` result) shaped to your schema—handy for **POCs** and **serverless** flows where you want to avoid HTML parsing. ([docs.firecrawl.dev][1])

---

## Approach B — Multi-page / domain-wide via `/v2/extract`

Use this when you need images from **many product pages** or across a **section/domain**. Provide `urls` (supports `/*`), and specify a **prompt + schema**; Firecrawl runs a job and returns an `id`. Poll status to get the final structured result. You can turn on `enableWebSearch` to enrich results beyond the supplied domain. ([docs.firecrawl.dev][2])

**Start job (cURL)**

```bash
curl -s -X POST https://api.firecrawl.dev/v2/extract \
  -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/catalog/*"],
    "prompt": "For each product page, extract all gallery images with alt text and mark the primary hero.",
    "schema": {
      "type": "object",
      "properties": {
        "pages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "url": {"type": "string"},
              "images": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {"src":{"type":"string"}, "alt":{"type":"string"}, "is_primary":{"type":"boolean"}},
                  "required": ["src"]
                }
              }
            },
            "required": ["url","images"]
          }
        }
      },
      "required": ["pages"]
    },
    "ignoreInvalidURLs": true,
    "scrapeOptions": { "onlyMainContent": false, "removeBase64Images": true }
  }'
```

**Poll for results**

```bash
curl -s -X GET https://api.firecrawl.dev/v2/extract/<JOB_ID> \
  -H "Authorization: Bearer $FIRECRAWL_API_KEY"
# => { "success": true, "status": "completed|processing|failed|cancelled", "data": {...}, "expiresAt": "...", "tokensUsed": 123 }
```

(`ignoreInvalidURLs` yields an `invalidURLs` list instead of failing the whole job.) ([docs.firecrawl.dev][3])

---

## Handling dynamic pages & heavy media

* **Actions**: For SPA/infinite scroll galleries, add `actions` like `wait`, `scroll`, or `click` before scraping. ([docs.firecrawl.dev][1])
* **Base64 images**: Kept **off** by default via `removeBase64Images: true` (helps avoid huge payloads). Turn it **off** only if you explicitly need inline data URIs. ([docs.firecrawl.dev][6])
* **Screenshots**: Add a `{ type: "screenshot", fullPage, quality }` format to capture a visual of the page (not a substitute for gallery images, but useful for QA). ([docs.firecrawl.dev][6])

---

## Self-hosting Firecrawl (use your own OpenAI key)

* Follow the self-host guide; copy the example `.env`.
* Add **`OPENAI_API_KEY`** to turn on LLM-dependent features (like `/extract` and JSON format extraction).
* Note: self-host **does not** include cloud-only “Fire-engine” (anti-bot/robustness) features; plan proxies for tough sites. ([docs.firecrawl.dev][5])

---

## Post-processing tips

* **Resolve relative URLs** against `metadata.sourceURL` (shown in `/scrape` response). ([docs.firecrawl.dev][6])
* **Deduplicate** (normalize querystrings, pick best from `srcset`).
* **Primary image**: schema can ask model to mark `is_primary`; validate with simple heuristics (largest resolution, hero container).
* **Rights & robots**: Respect site ToS, robots, and licensing for downloaded assets.

---

## Minimal Node & Python snippets (SDK)

**Node (Firecrawl JS SDK) – single page to rawHtml**

```ts
import Firecrawl from "@mendable/firecrawl-js";
const firecrawl = new Firecrawl({ apiKey: process.env.FIRECRAWL_API_KEY });

const doc = await firecrawl.scrape("https://example.com/product/123", {
  formats: ["rawHtml"],
  includeTags: ["img", ".product-gallery"],
  onlyMainContent: false,
  waitFor: 1500
});

console.log(doc.rawHtml.length, doc.metadata?.sourceURL);
```

(Advanced formats and options are the same as the HTTP examples above.) ([docs.firecrawl.dev][1])

**Python (Firecrawl SDK) – `/v2/extract` with schema**

```python
from firecrawl import Firecrawl
fc = Firecrawl(api_key="fc-...")

schema = {
  "type":"object",
  "properties":{
    "images":{"type":"array","items":{"type":"object","properties":{
      "src":{"type":"string"},"alt":{"type":"string"},"is_primary":{"type":"boolean"}
    },"required":["src"]}}
  },
  "required":["images"]
}

res = fc.extract(
  urls=["https://example.com/product/123"],
  prompt="Extract all product images with alt text and indicate the primary hero image.",
  schema=schema
)
print(res.data)
```

(Extract usage & job handling shown in the docs, including synchronous SDK behavior and start/poll methods.) ([docs.firecrawl.dev][2])

---

## Appendix: what the API returns (quick shapes)

* **`/v2/scrape`** → `{ success, data: { markdown?, html?, rawHtml?, screenshot?, links?, actions?, metadata? } }`. ([docs.firecrawl.dev][6])
* **`/v2/extract`** (start) → `{ success, id, invalidURLs? }`; **GET status** → `{ success, status, data, expiresAt, tokensUsed }`. ([docs.firecrawl.dev][7])

---

If you want, I can turn the JSON-extraction examples above into **LangChain `@tool`** wrappers so your ReACT agent can call “extract\_images(url)” on demand, with either the `/scrape` one-pass JSON format or the multi-page `/extract` job.

[1]: https://docs.firecrawl.dev/advanced-scraping-guide "Advanced Scraping Guide | Firecrawl"
[2]: https://docs.firecrawl.dev/features/extract "Extract | Firecrawl"
[3]: https://docs.firecrawl.dev/api-reference/v2-endpoint/extract "Extract - Firecrawl Docs"
[4]: https://docs.firecrawl.dev/api-reference/v2-introduction?utm_source=chatgpt.com "Introduction - Firecrawl Docs"
[5]: https://docs.firecrawl.dev/contributing/self-host "Self-hosting | Firecrawl"
[6]: https://docs.firecrawl.dev/api-reference/v2-endpoint/scrape "Scrape - Firecrawl Docs"
[7]: https://docs.firecrawl.dev/api-reference/endpoint/extract-get "Get Extract Status - Firecrawl Docs"
