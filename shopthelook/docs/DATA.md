# Data: sources, subsets, licenses

All counts below come from the prep scripts (`src/data/prepare_df2.py`,
`src/data/prepare_catalog.py`). The committed `splits/` folder lists the exact
images in every split, so anyone can rebuild the same subsets.

## 1. DeepFashion2 (detection + retrieval evaluation)

| | |
|---|---|
| What | 491k photos of clothing from shops and consumers, 13 garment categories, boxes + consumer-to-shop pair labels |
| Official access | Google Drive, with zips password-protected. The password comes from a request form (see the [official repo](https://github.com/switchablenorms/DeepFashion2)) |
| What we downloaded | An unofficial, unencrypted Hugging Face re-upload: [`sahirp/deepfashion2`](https://huggingface.co/datasets/sahirp/deepfashion2) |
| License | The official README gives no license text. Access goes through the authors' request form, so treat the data as **research / non-commercial use only**. The "MIT" tag on the HF re-upload was set by the uploader and is **not** authoritative. Anyone reproducing this should submit the official form. |

### Subsets used

| Split | Source | Images | How chosen | Used for |
|---|---|---|---|---|
| `train` | DF2 train (191,961 images) | 15,000 | uniform random sample, seed 42 (`splits/df2_train_subset.txt`) | YOLO fine-tuning |
| `det_val` | DF2 validation | 1,510 | all images of 100 random `pair_id`s | checkpoint selection during training |
| `det_test` | DF2 validation | 6,005 | all images of 400 *other* random `pair_id`s | **reported** detection metrics |
| retrieval | DF2 validation | 32,153 (all) | all items | consumer-to-shop Recall@k |

* Only the 15k-image subset was downloaded from the 10.6 GB `train.zip`. HTTP range requests fetch just those files (`src/data/remote_zip.py`).
* The det_val / det_test split is made **by `pair_id`**, so photos of the same garment never fall on both sides.
* Box counts: train 24,439 · det_val 2,530 · det_test 9,744 (`results/data/df2_class_counts.csv`).
* **Rare classes:** `short_sleeve_outwear` has 46 training / 44 test instances, and `sling` has
  137 / 35 (and 0 in det_val). Per-class metrics for these two are statistically noisy.
* Retrieval evaluation uses frozen, pretrained embedders that never see DF2 during training, so using the full validation set is still held out.

### Retrieval benchmark (DF2 validation, consumer-to-shop)

| | count |
|---|---|
| Queries: garments in consumer ("user") photos with `style > 0` | 12,377 |
| Gallery: all garments in shop photos | 36,961 |
| Correct match | same `pair_id` **and** same `style` (official DF2 definition) |

## 2. Product catalog (what the demo retrieves from)

| | |
|---|---|
| Original | Kaggle [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Param Aggarwal). The products come from Myntra. |
| License | **MIT** (as listed on Kaggle, for both the full and the "small" version). The product photos themselves belong to the retailer and brands. Use them for demos and research only, never commercially. |
| Downloaded | [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (metadata, 44,072 rows, 271 MB) + [`ceyda/fashion-products-small`](https://huggingface.co/datasets/ceyda/fashion-products-small) (same products at up to 512 px, 42,700 rows, 597 MB), joined on product id → 42,426 products |
| Kept | Topwear / Bottomwear / Dress whose `articleType` maps to a DF2 group → **17,330 products** |

### Catalog coverage by retrieval group

| group | products | share |
|---|---:|---:|
| top | 14,249 | 82.2% |
| trousers | 1,756 | 10.1% |
| shorts | 494 | 2.9% |
| dress | 437 | 2.5% |
| outerwear | 280 | 1.6% |
| skirt | 114 | 0.7% |

The catalog is heavily skewed towards tops. Skirt and outerwear matches come from a
small pool, which is a real product risk (see `docs/PRD.md`, catalog coverage).

Article types left out because DF2 has no counterpart: Dupatta (114), Suspenders (40),
Stockings (32), Tracksuits (29), Jumpsuit (16), Waistcoat (15), Swimwear (13),
Rompers (12), Salwar and Dupatta (7), Lehenga Choli (4), Belts (3).

### Category mapping (13 DF2 classes → 6 groups)

The catalog has no sleeve-length labels, so retrieval filters on a coarser group:

| group | DF2 classes | catalog articleTypes |
|---|---|---|
| top | short/long sleeve top, vest, sling | Tshirts, Shirts, Tops, Tunics, Sweaters, Sweatshirts, Kurtas, Kurtis |
| outerwear | short/long sleeve outwear | Jackets, Blazers, Shrug, Rain Jacket, Nehru Jackets |
| shorts | shorts | Shorts |
| trousers | trousers | Trousers, Jeans, Track Pants, Capris, Leggings, Jeggings, Tights, Patiala, Salwar, Churidar |
| skirt | skirt | Skirts |
| dress | short/long sleeve, vest, sling dress | Dresses |
