Create a publication-quality scientific visualization for Figure 2 of a high-level computational pathology / medical AI conference paper.

The figure must visualize the LEARNED TOP-K TOKEN SELECTION mechanism of a Vision Transformer model for HER2 grading from H&E-stained histopathology images.

The purpose of this figure is strictly to visualize how patch-token scores are distributed across an H&E image and which patch tokens are retained by the Top-K mechanism.

IMPORTANT:
This is a visualization of the model's learned token scores and selected tokens. Do NOT claim or visually imply that the selected regions are pathologically proven biomarkers, diagnostically important regions, HER2-positive regions, tumor regions, or expert-annotated regions. The figure must remain scientifically neutral.

Do NOT include IHC, IHC guidance, teacher networks, cross-attention, class-aware routing, ordinal-aware routing, or any other component not explicitly described below.

---

## FIGURE STRUCTURE

Create a compact horizontal multi-panel figure with FOUR ROWS, corresponding to four HER2 grades:

Row 1: HER2 0
Row 2: HER2 1+
Row 3: HER2 2+
Row 4: HER2 3+

Each row should contain THREE PANELS:

Panel A — Original H&E image
Panel B — Token-score visualization
Panel C — Top-K selected-token visualization

The four rows should use the SAME visual layout and identical spatial coordinate system.

---

## PANEL A — ORIGINAL H&E IMAGE

For each row, show a representative realistic H&E-stained breast histopathology image.

Label each row clearly:

“HER2 0”
“HER2 1+”
“HER2 2+”
“HER2 3+”

The H&E images should look like realistic histopathology rather than artificial microscopy illustrations.

Do not add annotations, arrows, tumor outlines, nuclei labels, or manually drawn pathology regions.

The images should represent the same type of 224 × 224 model input used by the architecture.

---

## PANEL B — TOKEN SCORE MAP

For each H&E image, overlay a clean 16 × 16 patch-token grid corresponding to the 256 Virchow2 patch tokens.

The grid must contain EXACTLY:

16 × 16 = 256 patch tokens.

Each grid cell represents one patch token.

Visualize the learned scalar token score produced by the model's:

Linear(1280 → 1)

scoring layer.

Use a continuous, perceptually clear heatmap representation to indicate relative token scores.

Higher-scoring tokens should be visually distinguishable from lower-scoring tokens.

The heatmap must be spatially aligned with the 16 × 16 token grid.

The underlying H&E morphology should remain visible so the viewer can understand the spatial correspondence between the token grid and the original image.

Add a small legend:

“Relative token score”

Do NOT label the score as:

* attention
* attention weight
* probability
* HER2 relevance
* tumor probability
* diagnostic importance

It is specifically a learned scalar token score.

---

## PANEL C — TOP-K SELECTION

Show the same 16 × 16 token grid.

Clearly indicate which tokens were selected by the Top-K mechanism.

Exactly:

K = 64

tokens must be selected from:

256

total patch tokens.

Therefore the visualization must represent:

64 / 256 selected tokens.

Use a clear but restrained visual distinction between selected and non-selected tokens.

For example:

* selected tokens: clearly outlined or highlighted cells
* non-selected tokens: subtle/light cells

The selected cells should correspond to the highest-scoring tokens from Panel B.

IMPORTANT:
The selected tokens must be spatially identical to the top-scoring locations in Panel B.

Do not randomly select locations.

Show a small label:

“Top-K selected tokens”
“K = 64”

If useful, also show:

“64 of 256”

---

## VISUAL CONNECTION BETWEEN PANELS

Within each row, use subtle arrows:

Original H&E
→
16 × 16 Patch Tokens
→
Learned Token Scores
→
Top-K = 64

The arrows must be thin, clean, and professional.

The figure should communicate the following process immediately:

H&E image
→ 256 patch tokens
→ scalar score for each token
→ rank tokens
→ retain highest-scoring 64 tokens

Do not introduce any additional processing step.

---

## SCIENTIFIC ACCURACY

The visualization must accurately represent the actual HER2Former implementation:

* Virchow2 produces 256 patch tokens.
* Each patch token has dimension 1280.
* A learnable Linear(1280 → 1) layer assigns one scalar score to each patch token.
* Top-K selection retains exactly 64 of the 256 patch tokens.
* Selection is based on the learned scalar scores.
* No attention weights are used for this selection.
* No IHC information is used.
* No class-conditioning is used.
* No ordinal-conditioning is used.
* No cross-attention is used.

Do not visualize the CLS token in this figure.
Do not visualize register tokens in this figure.

This figure is ONLY about the 256 patch tokens and Top-K selection.

---

## IMPORTANT INTERPRETATION RESTRICTION

The figure must NOT visually imply that the selected tokens have been validated by pathologists.

Do not add:
“clinically relevant regions”
“HER2-positive regions”
“diagnostic regions”
“tumor regions”
“biomarker regions”
or similar claims.

Use neutral terminology such as:

“Learned token score”
“Top-K selected tokens”
“Selected patch tokens”

The figure demonstrates model behavior, not pathological ground truth.

---

## VISUAL STYLE

Design the figure as a professional scientific figure for a strong machine learning / medical imaging / computational pathology conference.

Use a restrained academic visual language:

* clean vector-style layout
* realistic H&E imagery
* consistent panel dimensions
* precise alignment
* thin professional arrows
* subtle borders
* readable typography
* minimal decorative elements
* generous whitespace
* restrained academic color palette
* high contrast
* no unnecessary icons
* no 3D effects
* no glossy effects
* no cartoon elements
* no marketing-style graphics
* no excessive gradients

The figure should look like a figure from a high-quality peer-reviewed computational pathology paper.

---

## LAYOUT

Use a wide landscape composition suitable for a two-column conference paper.

Suggested top headers:

“Input H&E”
“Token Score Map”
“Top-K Selection”

Then four aligned rows:

HER2 0
[H&E] → [16 × 16 score map] → [Top-K = 64]

HER2 1+
[H&E] → [16 × 16 score map] → [Top-K = 64]

HER2 2+
[H&E] → [16 × 16 score map] → [Top-K = 64]

HER2 3+
[H&E] → [16 × 16 score map] → [Top-K = 64]

Make all three columns have identical dimensions across rows.

---

## TEXT AND TYPOGRAPHY

Use a clean professional academic sans-serif font.

All labels must be:

* correctly spelled
* horizontally aligned
* clearly readable
* concise
* free of unnecessary text

Do not put a large title inside the image.

Use only the labels necessary to explain the visualization.

---

## OUTPUT QUALITY

Generate an extremely high-resolution, publication-ready figure.

The figure must have:

* crisp text
* sharp patch grids
* perfectly aligned panels
* clean arrows
* no distorted labels
* no spelling errors
* no overlapping text
* no cropped panels
* no visual ambiguity about the 16 × 16 grid
* no ambiguity about selecting 64 of 256 tokens

The final figure should be suitable for inclusion as Figure 2 in a five-page, two-column scientific conference paper.
