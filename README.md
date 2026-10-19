# See-Sense: How a CNN "Sees"

- GitHub: https://github.com/reecegerhart/Cst435Topic3/tree/main
- Streamlit Cloud (UI): https://cst435topic3-rir3mms7vxalwudzcq8tds.streamlit.app/
- Render.com (API): https://cst435topic3.onrender.com/healthz
- Supabase project reference: cwojykvyxpwhywbhojvb (https://cwojykvyxpwhywbhojvb.supabase.co)
- Product presentation video(s): https://youtu.be/FzZsLuBnVVA
- Engineering report: seebelow 

## Pitch

See-Sense is an image-recognition demo for an AI literacy course. A visitor uploads an image, and a CNN returns a predicted class, top-5 probabilities, and a Grad-CAM overlay showing which pixels influenced the decision. A Recent Classifications tab reads from Supabase and links each prediction to the training run that served it. The product's purpose is pedagogical: visitors should leave understanding that a CNN does not see the way a human does.

We chose Fashion-MNIST over CIFAR-10 because its 28x28 grayscale images train quickly on free-tier hardware, which let us train both models under an identical compute budget and iterate often. The trade-off is that it is cleaner and easier than real photos, which the Model Card's limitations and the out-of-distribution demo make explicit.

## Architecture

Streamlit UI (thin client, no model code) -> FastAPI on Render (CNN, preprocessing, Grad-CAM) -> Supabase Postgres (runs, predictions, image_metadata). Only SHA-256 hashes, dimensions, and labels are stored. Raw images are never persisted.

## Engineering Report

### 1. Quantitative Analysis

Both models were trained on Fashion-MNIST for 10 epochs with batch size 64, learning rate 0.001, seed 42, and the AdamW optimizer. The baseline used no augmentation, batch normalization, dropout, or weight decay. The improved model added data augmentation, batch normalization, dropout of 0.3, weight decay of 0.01, and a cosine learning-rate schedule.

On the 10,000-image test set, the baseline reached 90.99% top-1 accuracy (test loss 0.2649, macro-F1 90.75%) and the improved model reached 89.33% (test loss 0.2828, macro-F1 89.35%). The added techniques did not improve overall performance in this experiment.

Figure 1 shows the training and validation curves. The baseline finished at 93.09% training and 91.42% validation accuracy, a gap of 1.67 points, so it shows only mild overfitting. The improved model finished at 87.27% training and 90.55% validation accuracy, with validation loss (0.2557) below training loss (0.3487). That pattern is expected when augmentation and dropout make the training task harder, and it indicates the regularization reduced fit to the training data without producing better test performance.

Figure 2 shows the confusion matrices and Table 1 the per-class precision and recall. Trouser, Bag, and Sandal were easiest for both models, with recall near or above 97%. Shirt was hardest: baseline recall was 61.1%, and the improved model raised it to 70.2%, but its shirt precision fell from 85.45% to 67.76%, meaning it labeled more other garments as shirts. The matrices show Shirt confused with T-shirt/top, and Pullover with Coat. The improved model also lost recall on T-shirt/top, Pullover, Dress, and Ankle boot, so regularization is not automatically an improvement. It depends on the dataset and architecture.

**Figure 1:** Training and validation loss and accuracy curves for both models. [INSERT IMAGE]

**Figure 2:** Confusion matrices for both models on the test set. [INSERT IMAGE]

**Table 1:** Per-class precision and recall on the test set.

| Class | Baseline precision | Baseline recall | Improved precision | Improved recall |
|---|---:|---:|---:|---:|
| T-shirt/top | 83.60% | 89.20% | 84.23% | 83.30% |
| Trouser | 98.70% | 98.90% | 98.59% | 97.90% |
| Pullover | 82.62% | 89.40% | 86.41% | 80.10% |
| Dress | 88.00% | 96.10% | 88.88% | 90.30% |
| Coat | 83.45% | 86.20% | 82.23% | 84.70% |
| Sandal | 98.29% | 97.70% | 98.08% | 97.30% |
| Shirt | 85.45% | 61.10% | 67.76% | 70.20% |
| Sneaker | 96.26% | 95.30% | 93.36% | 97.00% |
| Bag | 98.00% | 98.10% | 97.90% | 97.90% |
| Ankle boot | 95.61% | 97.90% | 96.73% | 94.60% |

### 2. Qualitative Analysis

[DESCRIBE what each model's Grad-CAM overlays highlight on clearly classified items, for example whether the heat falls on the garment's silhouette or spills onto the background, and how the improved model differs.]

Failure 1: a [TRUE LABEL] predicted as [PREDICTED LABEL] with [XX%] confidence by the [baseline/improved] model. The heatmap [WHAT IT SHOWED]. This is a [dataset / model] failure because [REASON FROM THE HEATMAP].

Failure 2: a [TRUE LABEL] predicted as [PREDICTED LABEL] with [XX%] confidence. The heatmap [WHAT IT SHOWED]. This is a [dataset / model] failure because [REASON FROM THE HEATMAP].

Out-of-distribution check: a photo of a [dog] was classified as [LABEL] with [XX%] confidence. Softmax always sums to 100% over the ten known classes, so a confident answer says nothing about whether the input belongs to any of them, which is why the interface labels the probabilities uncalibrated.

### 3. Ethical Reflection

A classifier of this family could cause harm in medical triage, where a confident wrong label could delay care, in content moderation, where systematic errors can silence particular groups, and in surveillance, where labels attached to people become accusations. The shared failure is a model trained on narrow data being applied to inputs it has never seen while reporting high confidence. Two safeguards a responsible team should require before shipping are an abstention threshold with out-of-distribution detection, so the system declines instead of guessing, and mandatory human review with an appeals path for any decision affecting a person, backed by per-subgroup performance audits.

Our design deliberately does not log raw images. Supabase stores only a SHA-256 hash, the dimensions, the predicted label, the top-5 probabilities, and the serving run. Uploads can contain faces, locations, or other personal information, and a public Recent Classifications wall would otherwise become a collection of strangers' photos. The hash still supports duplicate detection and traceability from a prediction to its model run, which is all the audit trail needs. Data that is never stored cannot be leaked or repurposed.

### 4. Worldview Reflection

If people are image-bearers (Genesis 1:27, ESV), then a system that sorts them carries moral weight that an accuracy score does not capture. I would refuse to build a version of this model that classifies people, such as inferring identity, emotion, or risk from faces or bodies for surveillance or screening, because it places automated power over individuals who cannot contest it. That refusal is part of the engineering. What a system ingests, logs, and is allowed to answer are design decisions with the same standing as the choice of loss function, and storing hashes instead of pixels is a small example of the architecture limiting what the system can do to people.

### 5. Pytest Tests

The suite in `tests/` covers a multipart upload smoke test for `/predict`, a shape test confirming the Grad-CAM overlay matches the input image dimensions, a regression test on a fixed reference image, and a Supabase round-trip test confirming `/predict` writes a `predictions` row linked to a `runs` row. Result of `pytest -q`: [N passed, N skipped].

### 6. Deployment

All three clouds are deployed, with links at the top of this README.

## Individual Contribution

Key points:
- Same as tasks identified I worked alone

Tasks identified in the project:
- Fork the template and choose the dataset
- Write and apply the SQL migration (runs, predictions, image_metadata)
- Train and persist the baseline CNN
- Train and persist the improved CNN
- Build the FastAPI endpoints (/predict, /runs, /predictions, /classes, /healthz, /version)
- Build the five Streamlit tabs
- Write the pytest suite
- Complete the Model Card
- Deploy all three clouds
- Record the video and write the engineering report