# PRD: Shop the Look (visual outfit search)

| | |
|---|---|
| Author | Product (portfolio project) |
| Status | Prototype built and evaluated offline · A/B test planned, not run |
| Related | `METRICS.md` · `AB_TEST.md` · `DATA.md` · `wireframes/shop_the_look_wireframe.html` |

## 1. Problem statement

Shoppers often find fashion **visually**: a street-style photo, an influencer post, a friend's
outfit, a film still. To buy something similar on a fashion app today, they have to turn the
picture into words ("olive cropped utility jacket with patch pockets") and hope the text search
understands them. This breaks down in three ways:

1. **Vocabulary gap.** Shoppers don't know the catalog's terms (shacket? co-ord? A-line?), so queries are vague or wrong.
2. **Attribute overload.** Colour, pattern, fit, sleeve and neckline are hard to express together, and filters only cover some of them.
3. **One item at a time.** An outfit is 2–4 garments. Text search makes the shopper repeat the whole effort for each one.

The result is long search sessions, many reformulations, drop-off, or buying the wrong thing and
returning it, which is costly in Indian fashion e-commerce where return rates are already high.

**Opportunity:** let the photo be the query. Detect each garment in the photo and show similar
products for each one, in one step.

## 2. Target user and persona

**Primary persona: Ananya, 24, Bengaluru, early-career professional**
* Shops fashion on her phone 2–3 times a month. Discovers styles on Instagram and saves screenshots "to find later".
* Pain: *"I know exactly what I want because I'm looking at it, but I can't find it by typing."* She gives up after 3–4 queries.
* Success looks like: screenshot → similar pieces at her price point in under a minute, for the whole outfit.

**Secondary persona: Rohan, 31, Pune, low-effort shopper**
* Doesn't follow fashion vocabulary. Wants "a jacket like the one in this photo" and is happy with *similar*, not *identical*.
* Uses the camera on a friend's outfit or a mannequin in a mall.

**Not targeted in v1:** stylists and resellers who need exact SKU matches, and shoppers looking for a specific branded item (text or barcode search already works for them).

## 3. User stories

| # | As a… | I want to… | So that… | Priority |
|---|---|---|---|---|
| U1 | shopper with a screenshot | upload it from my gallery | I don't have to describe the outfit | P0 |
| U2 | shopper | see each garment in the photo picked out separately | I can shop the top and the trousers independently | P0 |
| U3 | shopper | see ~5 similar products per garment, in the right category | I'm not shown shoes when I want a jacket | P0 |
| U4 | shopper | get results in about a second | it feels as fast as search | P0 |
| U5 | shopper | fix a wrong detection ("this is a jacket, not a top") | one mistake doesn't ruin the result | P1 |
| U6 | shopper | filter results by price, size availability and brand | I only see what I can actually buy | P1 |
| U7 | shopper | add the whole look to cart in one tap | I can buy the outfit together | P1 |
| U8 | shopper | take a photo with the camera | I can shop what I see in person | P2 |
| U9 | merchandiser | see which garments shoppers search for but we don't stock | I can fix catalog gaps | P2 |

## 4. Scope and non-goals

**In scope (v1)**
* Upload from gallery or camera, single image, app only.
* Detection of 13 garment categories (DeepFashion2 taxonomy), grouped into 6 shoppable groups: top, outerwear, shorts, trousers, skirt, dress.
* Top-5 similar in-stock products per detected garment, filtered by category, ranked by visual similarity.
* Basic corrections (change category, remove a box) and "add look to cart".

**Non-goals (v1)**
* Shoes, bags, jewellery and accessories (not in the detector's taxonomy; v2).
* Exact-product identification or brand logo recognition.
* Text + image combined queries ("this but in blue"). This is a strong v2 candidate.
* Virtual try-on, size recommendation, or anything that infers body shape from the photo.
* Personalised re-ranking (v2, once there is feedback data).
* Storing user photos server-side beyond the request (privacy by default).

## 5. Key flows

**Flow A: Screenshot to cart (happy path)**
1. Search bar → camera icon → "Choose from gallery" → picks an Instagram screenshot.
2. Loading skeleton (target p95 < 1.5 s) → photo with numbered boxes (1 Top, 2 Trousers).
3. Row 1: 5 similar tops. Row 2: 5 similar trousers. Each shows image, name, price and match score.
4. Taps a top → product page → size → add to cart. Or "Add both to cart" from the results page.

**Flow B: Wrong detection**
1. Box 1 is labelled "Top" but it's a jacket → taps "Not a top?" → picks "Jacket / outerwear".
2. Row 1 re-ranks within outerwear. The correction is logged (wrong-garment flag rate).

**Flow C: Nothing detected / no stock**
* No garment found → "Try a full-outfit photo in good light" with example thumbnails.
* Category detected but thin stock (e.g. skirts) → show what exists plus "similar dresses" as a labelled fallback, and log the gap for merchandising (U9).

Wireframes for Screens 1–2: `docs/wireframes/shop_the_look_wireframe.html`.

## 6. Solution overview (as built in the prototype)

```
photo ─▶ garment detector ─▶ crop each garment ─▶ image embedding ─▶ FAISS search within the
         (YOLOv8s-seg,         (box + 2% pad,       (FashionCLIP /      detected category's
          13 DF2 classes)       pad to square)        CLIP ViT-B/32)    catalog slice → top-5
```

* **Detection:** pretrained DeepFashion2 YOLOv8s-seg (third-party, Apache-2.0 repo, AGPL-3.0 Ultralytics terms). Evaluated by us on a held-out split.
* **Embedding:** CLIP and FashionCLIP compared against a colour-histogram baseline on consumer-to-shop retrieval. The winner is used in the demo.
* **Retrieval:** exact cosine search (FAISS `IndexFlatIP`), one sub-index per category group.
* **Catalog:** 17,330 apparel products (Kaggle Fashion Product Images; MIT license; products originally from Myntra).

## 7. Risks and mitigations

| Risk | Why it matters | Evidence from the prototype | Mitigation |
|---|---|---|---|
| **Wrong-category matches** | Showing trousers for a top destroys trust instantly | Without the category filter, a large share of top-5 results come from the wrong group (see §8). Rare classes like short-sleeve outerwear are detected poorly | Always filter by detected group; let users correct the category in one tap; hide low-confidence boxes (threshold 0.35) |
| **Detector misses or confuses garments** | No box means no recommendations for that item | Per-class AP ranges widely; the weakest classes have very few training examples | Manual box drawing (v1.1); targeted data collection for weak classes; monitor detection success rate by photo source |
| **Bias across skin tones and body types** | Detection or retrieval quality that differs by skin tone, body size, gender presentation or modest/ethnic wear is unfair and a brand risk | **Not measured yet.** DF2 has no skin-tone or body-type labels, and we don't infer these attributes from images | Before launch: build a consented, human-labelled evaluation set (skin tone via the Monk scale, body size, ethnic wear) and report detection recall and retrieval relevance per segment, with a parity threshold (e.g. no segment > 5 pp below the overall average). Add Indian ethnic wear (kurta, saree, lehenga) to the taxonomy, since it's missing today |
| **Catalog coverage** | Great matching is useless if the right kind of product isn't stocked | Our catalog is 82% tops. Only 114 skirts and 280 outerwear items, so skirt matches come from a tiny pool | Show coverage-aware fallbacks; report "searched but not stocked" to merchandising; in production use the full live catalog, filtered to in-stock sizes |
| **Latency** | Visual search that feels slow gets abandoned | Measured on a laptop (Apple M4): see §8. Production must hit p95 ≤ 1.5 s on server GPUs | GPU inference server; batch crops; cache catalog embeddings; downscale uploads on device; ANN index if the catalog grows past ~1M |
| **"Similar" ≠ "what I wanted" → returns** | Visually similar items can differ in fabric, fit or exact colour | Offline metrics can't measure this | Return rate of look-attributed orders is a guardrail with a kill threshold (see `AB_TEST.md`) |
| **Privacy** | Photos can contain faces, children, homes | — | Process in memory and don't store server-side by default; never use photos for training without explicit opt-in; blur faces in any logged samples |
| **IP and licensing** | Third-party weights and datasets carry terms | DF2 is research-only; Ultralytics models are AGPL-3.0 | Production needs a detector trained on owned data and an Ultralytics enterprise license (or a permissively licensed architecture) |

## 8. Prototype evidence (offline, measured)

All numbers come from `results/RESULTS.md`, generated from the evaluation outputs.

| Question | Evidence | What it means for the product |
|---|---|---|
| Can we find the garments? | Detector box **mAP@50 0.842** (official DF2 boxes) on 6,005 held-out photos. Trousers 0.972 and short-sleeve tops 0.964, but short-sleeve outerwear only **0.503** (44 test boxes) | Good enough for tops and bottoms. Jackets need data or a manual-box fallback |
| Does visual similarity find the right item? | Consumer-to-shop **Recall@10: 53.1% with FashionCLIP** vs 23.8% with general CLIP and 6.7% with a colour histogram (category-filtered, 12,377 queries vs 36,961 shop items) | A fashion-specific embedding more than doubles retrieval quality over general CLIP for no extra latency. Use FashionCLIP |
| Does the category filter matter? | Without it, **34.6%** of FashionCLIP's top-5 catalog results are a different garment type (73.2% for outerwear queries). Filtering also lifts consumer-to-shop R@10 from 47.8% to 53.1% | The filter is a P0 requirement, and so is the one-tap category correction |
| Where is quality weakest? | Trousers R@10 30.4% vs dresses 60.1%. Catalog search matches colour less often than a plain colour histogram (41.2% vs 54.6% same colour @5) | v1.1: re-rank with a colour term; add attribute filters (colour, fit) |
| Is it fast enough? | **Median 38.4 ms, p95 48.9 ms** end-to-end on an Apple M4 laptop (200 photos, excludes upload/decode) | Model compute is far inside the 1.5 s budget. In production the budget goes to network and upload |
| What broke? | The same garment was boxed twice ("shorts" + "skirt"). Fixed with class-agnostic NMS, which cut items per photo from 1.54 to 1.40 | One garment, one box. This is a product rule, not just a model setting |

## 9. Phased rollout plan

| Phase | Audience | Goal | Exit criteria |
|---|---|---|---|
| **0. Offline prototype** (this repo) | — | Prove detection + retrieval quality on public data | Detection mAP@50 ≥ 0.80 on held-out data; FashionCLIP or CLIP beats the colour baseline on Recall@k; latency measured ✔ |
| **1. Internal dogfood** | Employees, 2 weeks | Find UX breaks; verify logging (`source=shop_the_look`) | Crash rate < 0.5%; logging verified; ≥ 200 labelled "was this relevant?" ratings |
| **2. Bias and quality review** | Offline, consented eval set | Check fairness across skin tone, body size and ethnic wear | No segment more than 5 pp below average on detection recall or rated relevance |
| **3. A/B test** | 50/50 of eligible users, 14 days | Causal lift in ATC conversion (`AB_TEST.md`) | Ship / iterate / kill rules in `AB_TEST.md` §7 |
| **4. Ramp** | 10% → 50% → 100% over 3 weeks, 5% long-term holdout | Confirm the effect holds and returns stay flat | Holdout readout at 8 weeks, including 30-day return rates |
| **5. v2** | — | Accessories and shoes, text+image queries ("this in blue"), personalised ranking, "complete the look" | Separate PRDs |

## 10. Open questions

1. Is "similar" enough, or do shoppers expect exact items? (A dogfood survey and recommendation CTR will tell us.)
2. Should the entry point also live on the home feed, or only in search? (Test as a follow-up variant.)
3. How should we rank similar items that differ in price? (Visual-only today. Blending with price affinity is v2.)
