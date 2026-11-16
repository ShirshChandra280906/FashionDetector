# ShopTheLook: Metrics framework

How we judge whether Shop the Look is working, from the North Star down to
the model metrics that drive it. All metrics are **defined** here. No
production data exists for this prototype. Offline model numbers come from
`results/`, and A/B planning numbers are assumptions (see `AB_TEST.md`).

## 1. North Star

**Add-to-cart rate from look-based recommendations**

> Of all Shop-the-Look sessions, the share where the shopper adds at least one recommended item to cart within that session.

| | |
|---|---|
| Formula | `# look sessions with ≥1 ATC from a recommended item  /  # look sessions with ≥1 garment detected` |
| Unit | session (one photo upload, including re-crops and refinements of that photo) |
| Why this metric | It captures the whole promise: we understood the photo (detection), we found something similar enough (retrieval), and it was good enough to want (catalog + price). Clicks alone can be curiosity, and orders lag and depend on checkout and payment. ATC is the earliest strong signal of intent that the feature controls. |
| Why not GMV | GMV is driven by price mix and promotions, which the feature doesn't control, and it's too noisy for a 2-week test. We track it as a secondary outcome. |
| Caveat | It is only defined for users who use the feature, so it **can't be the A/B primary metric** (control has no look sessions). The A/B test uses user-level ATC conversion across both arms (see `AB_TEST.md`). |

## 2. Supporting (driver) metrics

These explain *why* the North Star moves, ordered along the funnel:

| Funnel step | Metric | Definition | What it diagnoses |
|---|---|---|---|
| Discover | Entry-point CTR | taps on "Shop the Look" camera icon / users who saw it | placement, icon clarity |
| Upload | Upload completion | photos submitted / entry-point taps | permissions friction, gallery UX |
| Understand | Detection success rate | uploads with ≥1 garment detected / uploads | detector recall on real photos, photo quality |
| Understand | Wrong-garment flag rate | "not this item" taps / detected items | detector precision, class confusion |
| Match | **Recommendation CTR** | product taps / recommendation rows shown | retrieval relevance |
| Match | Mean reciprocal rank of first tap | 1/rank of first tapped product per row | ranking quality, a live proxy for offline Recall@k |
| Convert | **Search-to-cart conversion** | sessions with ATC / sessions (Shop the Look vs text search, side by side) | is visual search a better path to cart than typing? |
| Convert | **Time to first add-to-cart** | median seconds from session start to first ATC | speed promise: "find it faster than text search" |
| Convert | Orders per look session | orders containing a recommended item / look sessions | downstream conversion |

## 3. Guardrail metrics (must not get worse)

| Guardrail | Definition | Threshold (non-inferiority) | Why |
|---|---|---|---|
| **Return rate of items bought via Shop the Look** | returned units / delivered units, for orders attributed to look recommendations, measured 30 days after delivery | ≤ +1.0 pp vs. category-matched control orders | "Looks similar" ≠ "is what I wanted". Wrong colour, fit or category shows up as returns, which cost the business more than a lost sale. |
| **p95 end-to-end latency** | upload received → results rendered (server side) | p95 ≤ 1.5 s (server); alert at 2 s | slow visual search loses users. The prototype's measured latency is in `results/latency/`. |
| **Bounce rate** | sessions with no further interaction after results load | ≤ +2 pp vs text-search sessions | bad matches make users leave the app entirely |
| App crash / error rate | failed uploads + crashes / uploads | ≤ 0.5% | camera/gallery permissions and large images |
| Overall ATC conversion (treatment arm) | user-level ATC across all surfaces | not significantly negative | cannibalisation: users diverted from a better path |

## 4. Offline model metrics (what we can measure today)

These are the levers we can tune before any user sees the feature. Each maps to a funnel step above.

| Offline metric | Measured on | Funnel step it predicts | File |
|---|---|---|---|
| Detection mAP@50, mAP@50-95, per-class AP | DF2 det_test (6,005 held-out photos) | detection success, wrong-garment flags | `results/detection/det_test_metrics.json` |
| Consumer-to-shop Recall@1/5/10 | DF2 validation (12,377 consumer garments vs 36,961 shop garments) | recommendation CTR, MRR | `results/retrieval/df2_c2s_summary.csv` |
| Wrong-category rate (unfiltered) | DF2 consumer garments vs our catalog | wrong-category matches, returns | `results/retrieval/catalog_wrong_category_unfiltered.csv` |
| Same-articleType / same-colour @k | catalog leave-one-out | perceived relevance | `results/retrieval/catalog_leave_one_out.csv` |
| Median / p95 latency | 200 held-out photos, Apple M4 | latency guardrail | `results/latency/` |

**Offline→online link we will check once live:** does a model with higher Recall@5 actually
raise recommendation CTR? If not, the offline benchmark (consumer-to-shop pairs, which reward
*exact* item matches) is the wrong proxy for a shopper who is happy with *similar* items, and we
should switch to human-rated relevance.

## 5. Segments to cut every metric by

New vs returning users · app vs mobile web · photo source (camera vs gallery vs screenshot) ·
garment group (top / bottom / dress / outerwear) · gender of catalog section · price band ·
city tier · device class (low-end Android latency). The bias review cuts by
skin tone and body type: these need consented, human-labelled evaluation sets, because we
never infer them from user photos (see PRD §7).
