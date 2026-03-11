# CMPE 597 Spring 2026 Term Project 

## MemeCap: Cross-Modal Retrieval, Caption Classification, and Sentiment Analysis

## 1. Project Overview

This project uses the MemeCap dataset, which contains meme images alongside meme captions, image captions, titles, metaphor annotations, and metadata. The overall goal is to study how well vision-language models understand memes, especially their metaphorical meaning.

This repository currently focuses on Task 2.1: Cross-modal Retrieval, where the objective is to retrieve the correct meme caption from the full set of candidate captions in the test dataset given:

* **Type 1 Input:** The meme image only. 
* **Type 2 Input:** The meme image together with its title. 

## 2. Dataset

The dataset is sourced from the official MemeCap repository.

### Dataset Splits
* **Training Set:** 5,823 memes
* **Test Set:** 559 memes

### Files & Directories
* `data/memes-trainval.json`: Training and validation annotations.
* `data/memes-test.json`: Test annotations.
* `data/memes/`: Directory containing the image files.

### Key Sample Fields
* `img_fname`: Image filename.
* `title`: Reddit post title of the meme.
* `meme_captions`: Intended meme meaning.
* `img_captions`: Literal description of the image.
* `metaphors`: Visual metaphor annotations.
* `post_id`: Sample identifier.

---

## 3. Repository Structure

```text
project/
├── data/
│   ├── memes-test.json
│   ├── memes-trainval.json
│   └── memes/
├── zero_shot/
│   ├── evaluation.py
│   ├── common.py
│   ├── openclip_backend.py
│   ├── siglip_backend.py
│   └── blip_reranker.py
├── outputs/
└── README.md
```

## 4. Task 2.1: Cross-Modal Retrieval

### Task Definition

The goal is to retrieve the correct meme caption for a given meme query. The system leverages a shared latent space where the similarity between a meme and its correct caption is higher than the similarity between the meme and unrelated captions.

For Type 2 Input, a zero-shot fusion strategy is used to combine the image and title into a single query representation:

$$q=\text{normalize}(\alpha \cdot e_{\text{image}}+(1-\alpha)\cdot e_{\text{title}})$$

*(Note: $\alpha = 0.7$ unless otherwise stated).*

### Evaluation Strategy

This task is formulated as an image-to-text retrieval problem. For each query meme, the model ranks all candidate meme captions in the test set according to similarity in a shared embedding space. This is a full gallery retrieval evaluation, meaning each query is ranked against all test captions, not against a small sampled subset.

#### BLIP Reranking

For experiments with reranking, the retrieval process is performed in two stages:

1. A dual-encoder model (OpenCLIP or SigLIP) retrieves the top-K candidate captions using embedding similarity.
2. A BLIP Image-Text Matching (ITM) model scores each candidate image–caption pair.
3. The final score is computed as a weighted combination of the base retrieval score and the BLIP score:

$$
S_{final} = \lambda S_{base} + (1-\lambda) S_{BLIP}
$$

where $\lambda = 0.5$ in our experiments.

#### Evaluation Protocol

1. Build the query embedding (Image only for Type 1; Fused Image + Title for Type 2).
2. Encode all candidate meme captions in the test set.
3. Compute similarity scores between the query and every candidate caption.

Similarity between the query embedding and caption embeddings is computed using cosine similarity in the shared embedding space:

$$
S(q, c) = q^\top c
$$

where both embeddings are L2-normalized.

4. Rank all candidate captions.
5. Measure whether the correct caption appears in the top retrieved positions.

#### Selected Metrics

**Primary Metrics:**
* **Recall@1 (R@1):** Percentage of queries where the correct caption is ranked first.
* **Recall@5 (R@5):** Percentage of queries where the correct caption appears within the top 5 retrieved captions.
  * *Pros:* Easy to interpret, directly reflects retrieval quality, and matches the task objective.
  * *Cons:* Coarse metric; gives no partial credit when the correct caption is just outside the top-K.

**Supplementary Metric:**
* **Mean Reciprocal Rank (MRR):** Measures how early the correct caption appears in the ranking (Average of $1/r$ over all queries).
  * *Pros:* Highly sensitive to ranking quality; better reflects whether the correct caption appears very early.
  * *Cons:* Focuses only on the first relevant item.

### Pretrained Architectures (Zero-Shot)

We evaluate pretrained cross-modal retrieval architectures in a zero-shot setting to test how well they transfer to the meme-caption retrieval problem.

1. **OpenCLIP (ViT-L/14):** Used as the main zero-shot retrieval baseline. Selected for its strong open-source image-text retrieval capabilities and efficiency for full-gallery ranking.
2. **SigLIP / SigLIP2:** Included as a strong alternative vision-language model family. Useful for comparison against standard CLIP-style contrastive training.
3. **BLIP Retrieval / BLIP ITM:** Used specifically as a reranker for top candidates retrieved from faster dual-encoder models to improve final retrieval quality.

---

## 5. Execution Commands

### OpenCLIP

**Type 1 Input:**
```bash
python zero_shot/evaluation.py \
  --data_dir data \
  --image_root data/memes \
  --model_family openclip \
  --openclip_model_name ViT-L-14 \
  --openclip_pretrained laion2b_s32b_b82k \
  --input_type type1 \
  --alpha 0.7 \
  --output_dir outputs
```

**Type 2 Input:**
```bash
python zero_shot/evaluation.py \
  --data_dir data \
  --image_root data/memes \
  --model_family openclip \
  --openclip_model_name ViT-L-14 \
  --openclip_pretrained laion2b_s32b_b82k \
  --input_type type2 \
  --alpha 0.7 \
  --output_dir outputs
```

**Type 2 Input + BLIP Reranking:**
```bash
python zero_shot/evaluation.py \
  --data_dir data \
  --image_root data/memes \
  --model_family openclip \
  --openclip_model_name ViT-L-14 \
  --openclip_pretrained laion2b_s32b_b82k \
  --input_type type2 \
  --alpha 0.7 \
  --use_blip_reranker \
  --blip_checkpoint Salesforce/blip-itm-base-coco \
  --blip_top_k 50 \
  --blip_blend_lambda 0.5 \
  --output_dir outputs
```

### SigLIP

**Type 1 Input:**
```bash
python zero_shot/evaluation.py \
  --data_dir data \
  --image_root data/memes \
  --model_family siglip \
  --siglip_checkpoint google/siglip2-large-patch16-384 \
  --input_type type1 \
  --alpha 0.7 \
  --batch_size 8 \
  --text_batch_size 16 \
  --output_dir outputs
```

**Type 2 Input:**
```bash
python zero_shot/evaluation.py \
  --data_dir data \
  --image_root data/memes \
  --model_family siglip \
  --siglip_checkpoint google/siglip2-large-patch16-384 \
  --input_type type2 \
  --alpha 0.7 \
  --batch_size 8 \
  --text_batch_size 16 \
  --output_dir outputs
```

---

## 6. Results

| Model | Input Type | R@1 | R@5 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| OpenCLIP (ViT-L/14) | Type 1 | 60.29 | 74.78 | 67.51 |
| OpenCLIP (ViT-L/14) | Type 2 | 56.71 | 71.38 | 63.81 |
| OpenCLIP + BLIP | Type 2 | 68.16 | 78.89 | 73.16 |
| SigLIP2 | Type 1 | 54.74 | 70.84 | 62.54 |
| SigLIP2 | Type 2 | 23.43 | 38.28 | 31.50 |

---

## 7. Project Roadmap

- [ ] **Task 2.1.c:** Custom Architecture Implementation
- [ ] **Task 2.1.d:** Finetuning Experiments
- [ ] **Task 2.2:** Literal vs. Metaphorical Caption Classification
- [ ] **Task 2.3:** Meme Sentiment Classification
- [ ] **Future Work**
- [ ] **References**