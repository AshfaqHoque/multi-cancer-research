# Notebook audit and VJCS submission plan

## Confirmed from uploaded notebook

- Title: "Multi-Cancer Image Classification Using a Hybrid CNN–Mamba Architecture with CBAM, DropBlock, and Grad-CAM++".
- Original dataset downloaded with `kagglehub.dataset_download("obulisainaren/multi-cancer")`.
- `MultiCancerDataset` walks subdirectories and assembles per-folder subclass labels; `exclude_types=["oral cancer"]`.
- Model configured for 24 output classes. Public Kaggle dataset describes 130,000 images in 26 subclass folders, with 10,000 oral-cancer images (2 subclasses); implies nominal 120,000 and 24 subclasses after exclusion, **provided the downloaded version matches**.
- Random split: approximately 70/20/10; seed 42, but not group-aware or stratified.
- Training transforms: 224x224, horizontal flip (p=0.5), rotation +/-15 degrees, ImageNet normalization. Validation/test: deterministic resize + normalization.
- Architecture: convolutions 3->32->64->128->256 with BN, ReLU, max pooling; CBAM stages 2 and 4; DropBlock(p=0.1, b=5) and (p=0.2, b=7); Mamba blocks after stages 2 and 4 with d_model=64/256, d_state=16, d_conv=4, expand=2; GAP, dropout=0.5, Linear(256,24).
- Batch size 32; Adam lr 0.001; cross entropy; training method supports checkpointing and early stopping (default patience 3).
- Evaluation code computes accuracy, weighted precision/recall/F1, confusion matrix, weighted one-vs-rest ROC-AUC, per-class ROC, and Grad-CAM++-style maps for 20 randomly selected test images.

## Critical gaps and limitations

1. Only four code cells have execution counts. The last saved execution attempts show failed package installation (`torch==2.4.0` with cu121 was not found; `causal-conv1d` installation failed) and an unsuccessful `mamba_ssm` import. There are **no persisted experiment outputs or training curves**.
2. Training invocation is commented out. Another dataset holds the training history and an external Kaggle model folder holds the checkpoint. Neither file is included in the uploaded notebook; test performance cannot be verified.
3. The current image-level random split offers no guarantee against multiple frames from one patient or near-identical images being spread across train/val/test.
4. A checkpoint trained on a different split could have seen current "test" images. Reconstruct split/checkpoint provenance before evaluation.
5. No strong baseline, controlled ablation suite, multi-seed variability, external validation, hardware description, or training timing is present.
6. The dataset combines image domains and includes labels that may be normal/nonmalignant. Phrase task as **multi-category dataset image classification**, not a validated clinical multi-cancer diagnostic system.
7. Grad-CAM++-style visualization is hooked on the second Mamba unit, not a conventional last convolution. Validate explanation implementation, display the original labels, test explanatory faithfulness, and do not equate heatmaps with pathology localization.
8. Mamba block has no explicit top-level residual; don't describe it as a residual architecture.

## Recommended experiments (priority order)

1. **Reproducibility + test leakage**: exact dataset source/version, image hashes, group metadata, manifest, model-checkpoint provenance, environment lockfile.
2. **Credible primary result**: train the full model end-to-end, save history, evaluation predictions and probabilities, 24-class labels, and full test report with accuracy, macro-F1 and weighted AUC.
3. **Comparisons**: same backbone without extensions, ResNet-50, EfficientNet (and optionally ViT) under matched splits/training budgets.
4. **Ablations**: CNN only; +CBAM; +DropBlock; +Mamba; Mamba+CBAM; full model. Three independent random seeds recommended.
5. **Statistical robustness**: confidence intervals; per-source subset metrics; group-/source-aware testing where possible.
6. **Interpretability**: stratified correct/incorrect visual examples; attention faithfulness/perturbation checks; annotation-based localization only if real masks exist.
7. **Writing finalization**: fill result tables and figures; revise Abstract/Discussion/Conclusion using actual results; final author/declaration text; publisher class/style and submission checklist.

## Closest VJCS reference papers identified

- **Luong et al. (2025)**. *An Approach for Breast Cancer X-Ray Images Classification Based on Vision Transformer.* VJCS 12(3), 329–347. DOI `10.1142/S2196888824500210`. Particularly close on cancer image classification, architecture comparison and deep learning. **Full-text headings could not be independently verified in this session**; do not present any guessed section list as exact.
- **Pham et al. (2023)**. *Chest X-Rays Abnormalities Localization and Classification Using an Ensemble Framework of Deep Convolutional Neural Networks.* VJCS 10(1), 55–73. DOI `10.1142/S2196888822500348`. Particularly close on medical-image system architecture, data preparation and comparative experiments. The exposed full-text extract explicitly confirms **1. Introduction** and **4. Experiments**, with **4.1 Dataset** and **4.2 Data cleaning and preprocessing**. Other exact top-level headings were not independently available.

**Recommended paper architecture in this draft (our own, not a copy of either paper):**

1. Introduction
2. Related Work
3. Materials and Methods
4. Experimental Design and Reproducibility
5. Results
6. Ablation Study and Explainability
7. Discussion
8. Conclusion and Future Work

Additional: Abstract, Keywords, Declarations, References, Appendix. This meets normal research-paper conventions while leaving room for the official publisher template.

## Ranking and indexing snapshot (November 2026)

- Journal: Vietnam Journal of Computer Science, World Scientific; eISSN 2196-8896.
- SCImago 2025: **Q3**, SJR **0.308** (secondary pages reporting 2025 SCImago data).
- Scopus CiteScore 2025: **4.0**, **Q2** in several assigned CS categories (different methodology from SCImago quartiles).
- WoS: ESCI; external third-party index listing reports JIF approximately **0.8** and JCR **Q4**; verify directly through Clarivate before formal reporting.
- DOAJ lists journal as open access, no APC, with anonymous peer review and plagiarism screening; average submission-to-publication ~13 weeks reported, not guaranteed.
- Author instructions: get and verify directly from World Scientific before submitting. The website blocked automated text access during this session, so exact formatting claims are not guaranteed.

## Research links

- VJCS official: https://www.worldscientific.com/worldscinet/vjcs
- DOAJ: https://doaj.org/toc/2196-8896
- Cancer ViT paper: https://doi.org/10.1142/S2196888824500210
- Chest X-ray hybrid paper: https://doi.org/10.1142/S2196888822500348
- Dataset: https://doi.org/10.34740/KAGGLE/DSV/3415848
- Scopus/SCImago context: https://journalsuggester.com/journal/21101060452
- 2025 CiteScore context: https://journalindexes.com/vietnam-journal-of-computer-science-21968888/
