# Seismic Exploration Copilot

**APGCE 2026 | Geoscience + AI MVP**  

**Status:** Working prototype / research demonstration; not a validated prospect evaluation system.

## Overview

Seismic Exploration Copilot brings together seismic interpretation, elastic attributes, machine-learning property prediction, and conversational AI in a Streamlit interface. The MVP is designed to help geoscientists move from seismic imagery and sampled attributes to **explainable, reviewable exploration hypotheses**.

The current app focuses on four experiences:

1. **Dr. Geomorph — geobody interpretation:** interpret an RGB spectral-decomposition time slice with an APGCE-hosted Gemma model; describe candidate depositional features and display approximate polygon overlays.
2. **Reservoir screening:** rank samples in a seismic-attribute CSV using sand probability, predicted porosity, Mu-Rho, and Vp/Vs; inspect the highest-ranked locations on real SEG-Y sections.
3. **Exploration chatbot:** ask follow-up questions grounded in the current geomorphology report and screening results.
4. **Processing workflow controls:** demonstrate where spectral-decomposition generation, PHIT prediction, and Mu-Rho calculation would be launched. These controls are **MVP placeholders** unless separately wired to processing functions.

## System architecture

```text
SEG-Y stacks + inversion volumes         Well LAS / trajectories / checkshots
                |                                     |
                +--------------+----------------------+
                               |
                    Data QC / alignment
                               |
             +-----------------+-------------------+
             |                                     |
    Spectral decomposition                 Sampled attribute CSV
      RGB time-slice PNGs                   + PHIT_pred, mu_rho
             |                                     |
       Dr. Geomorph                         Reservoir screening
       (Gemma via vLLM)                    (transparent heuristic)
             |                                     |
    JSON + polygon overlays                 Ranked candidates
             |                                     |
             +---------------+---------------------+
                             |
                      Streamlit app
                             |
                  Context-aware Gemma chat
```

## Features

### 1. Dr. Geomorph — RGB spectral geomorphology

- Upload a spectral-decomposition image (`.png`, `.jpg`, `.jpeg`, `.webp`).
- Read the time slice's **TWT** from its filename when present, or enter it manually.
- Configure red, green, and blue spectral frequencies; example mapping **15 / 30 / 60 Hz**.
- Run simple image measurements without an LLM.
- Ask Gemma to describe observable geometry before interpreting channels, bars, lobes, faults, mass-transport deposits, carbonate features, artefacts, and other candidates.
- Generate a structured report with depositional-environment hypotheses, possible indirect hydrocarbon indicators, uncertainty, false-positive risks, and recommended validation.
- Draw proposed feature polygons on the **original color image** and export the report JSON and overlay PNG.
- Filter implausibly large model-proposed polygons.

**Polygon coordinates:** normalized image-space coordinates on a **0–1000** grid. They are **not** automatically georeferenced to inline/crossline or map coordinates.

**Experimental enhancement:** backend-only grayscale, multiscale ridge detection and contour extraction was developed as an option for refining channel-like polygons while retaining RGB images for display and Gemma interpretation. It should be tested on clean slices before being treated as an integrated, reliable feature detector. Bright ridges do not necessarily trace full geological body boundaries.

### 2. Reservoir screening

Screen candidate samples from an uploaded CSV. The latest proposed scoring logic uses:

| Attribute | Preferred direction in this **heuristic** | Weight |
|---|---|---:|
| `TotalSandProb` | Higher | 40% |
| `PHIT_pred` | Higher | 30% |
| `mu_rho` | Lower, relative to qualifying samples | 15% |
| `VpVs` | Lower, relative to qualifying samples | 15% |

**Default screening thresholds:** `TotalSandProb >= 0.65` and `PHIT_pred >= 0.10`.

```text
screening_score = 0.40 * TotalSandProb
                + 0.30 * PHIT_pred
                + 0.15 * mu_rho_score
                + 0.15 * vpvs_score
```

`mu_rho_score` and `vpvs_score` are inverse percentile ranks within the qualifying dataset. This is a **relative prioritization score**, **not** a hydrocarbon probability or a calibrated geological classifier. Confirm that your local `reservoir_screening.py` contains this latest four-attribute scoring version before presenting these weights as the active implementation.

The app displays a ranked table, CSV download, and the **top three** candidate locations on selectable SEG-Y sections: **VpVs, TotalSandProb, PHIT_pred**. Each selected attribute has its own color map. The section viewer uses inline/crossline trace headers and TWT sample coordinates, and treats zero-filled inversion samples as no-data.

### 3. Context-aware exploration chatbot

The app uses the same Gemma endpoint for follow-up questions. Chat context includes the current Dr. Geomorph interpretation and reservoir screening summary/top-ranked rows where available. Conversation history is maintained in Streamlit session state.

Example questions:

- Why is Rank 1 favored over Rank 2?
- Which detected shapes are most consistent with channels?
- What other interpretations could explain this spectral response?
- What evidence is still needed to distinguish gas-bearing from water-bearing sand?

**Scope:** the chatbot reasons over **available reports and ranked data**. It does not independently read the entire seismic cube, prove hydrocarbon charge, or verify every claim against a well.

### 4. Processing workflow buttons

The UI includes buttons for **Generate spectral decomposition** and **Run PHIT prediction**. A **Calculate Mu-Rho** demonstration button can also display the elastic formula using `st.latex()`:

```python
st.latex(r'\mu\rho = Z_S^2 = (V_S\rho)^2')
```

These are explanatory or placeholder controls in the MVP; the underlying notebook processing is separate unless explicitly connected.

## Data and preparation completed during the geohackathon

### Seismic inputs

The APGCE challenge data includes:

- **Stacks:** Full, Near, Mid, Far, UltraFar.
- **Inversion:** P-impedance, S-impedance, Vp/Vs, Total Sand Probability.
- **Wells:** LAS curves including VSH, PHIT, SWT and oil/gas indicators; deviation data and selected checkshots.


SEG-Y geometry is not consistently recognized as structured by `segyio`; working readers use:

```python
with segyio.open(path, 'r', ignore_geometry=True) as f:
    ...
```

Trace locations are reconstructed from `INLINE_3D` and `CROSSLINE_3D` headers. QC covered available time ranges, missing traces, zero-filled inversion intervals, distributions, and outliers before normalization.

### Spectral decomposition

- Generated spectral-amplitude slices and combined low/mid/high frequencies into RGB displays.
- Applied percentile-based normalization and gamma adjustments for more interpretable image contrast.
- Exported slices at configurable TWT intervals (for example, every 100 ms).
- Experimented with rotation, image sizing, black backgrounds, and removing well labels/plot decorations.
- Created a notebook-friendly OpenCV approach to turn TWT-named image sequences into an MP4, sorted from **highest to lowest TWT**.

### Well and seismic property modeling

- Extracted stack/inversion attributes along well trajectories and aligned them with interpreted LAS curves.
- Experimented with XGBoost for **PHIT** prediction using seismic attributes.
- Applied a trained/exported model to a separate attribute dataset to produce `PHIT_pred`.
- Exported predicted values into a SEG-Y based on a template volume and inline/crossline/TWT matching.
- Investigated unusually perfect validation scores and potential leakage/split problems; **a 1.0 test score is not evidence of generalization** without independent well-based validation.

**Important export limitation:** predictions from the sampled CSV are spatially and temporally sparse. Writing only the exact predicted TWT samples into a full-resolution SEG-Y produces visible horizontal stripes/blank samples. Interpolation **within a trace** can fill time gaps between predictions, but it **does not** fill missing inline/crossline traces. Full-coverage PHIT requires full-grid inference or a separately justified spatial reconstruction.

### Sampled CSV

A 10%-style trace sample was used to keep the MVP manageable. The sampled table includes fields such as:

```text
Inline, Crossline, TWT, NEAR, MID, FAR, ULTRAFAR, AMP,
P_Impedance, S_Impedance, VpVs, TotalSandProb,
PHIT_pred, mu_rho
```

Optional `lambda_rho` can be retained for analysis. Sampling density and valid-data coverage should be checked for each export; the sampled table must not be presented as a continuous 3D body model.

### Elastic attributes

Given P-impedance `Z_P` and S-impedance `Z_S`:

```text
Mu-Rho     = Z_S²
Lambda-Rho = Z_P² - 2 Z_S²
```

These relations assume consistent physical impedance units. The user-facing MVP now expects `mu_rho` to be precomputed in the CSV rather than requiring the previous separate LMR calculation workflow.


### Requirements

- Python 3.10+ (the GeoHackathon work used Python 3.12).
- Packages: `streamlit`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `opencv-python`, `scipy`, `segyio`, `openai`.
- For notebook modeling/exports: `xgboost`, `scikit-learn`, `lasio`, `tqdm`, `jupyter` as applicable.
- Network access to the **APGCE-hosted vLLM/Gemma service**. No public OpenAI API key is required for this deployment.

```bash
python -m pip install streamlit numpy pandas matplotlib pillow opencv-python scipy segyio openai
```

The app imports `apgce_client.py`, which provides `MODEL`, `BASE_URL`, and `get_client()`. Configure it for your cluster's OpenAI-compatible vLLM endpoint. The model used in development was `google/gemma-4-26B-A4B-it`.

From the directory containing the entrypoint, run:

```bash
streamlit run app.py
```

If your entrypoint is under `notebooks/`, instead run:

```bash
streamlit run notebooks/app.py
```

**Path note:** `seismic_sections.py` contains environment-specific paths to the APGCE SEG-Y volumes and `outputs/PHIT_pred.sgy`. Adjust these paths for the machine where Streamlit runs. The SEG-Y files themselves are not distributed with this README.

## Demo walkthrough

1. Launch Streamlit and open **Dr. Geomorph**. Upload an RGB spectral-decomposition image; verify TWT and frequency mapping; click **Interpret with Gemma**.
2. Inspect the summary, feature list, uncertainty, and original-color polygon overlay. Download JSON/PNG if needed.
3. Open **Reservoir Screening** and upload the seismic attribute CSV with `TotalSandProb`, `PHIT_pred`, `mu_rho`, and `VpVs`.
4. Click **Run reservoir screening**. Review the ranked table and top-three candidates on the three attribute sections.
5. Use the **Chatbot** to ask why features or candidate ranks were proposed and what evidence is missing.
6. Show the **Processing Workflows** buttons as the planned integration points, not as completed automated processing.

## Limitations and scientific safeguards

- **No confirmed hydrocarbons:** spectral color, low Vp/Vs, low Mu-Rho, porosity prediction, and sand probability are indirect indicators and can be misleading without calibration.
- **Sample-level screening only:** top-ranked CSV rows do not establish laterally continuous reservoir bodies, connectivity, trap, seal, charge, or economic viability.
- **Model polygons are proposals:** pixel-space outlines can include plot artifacts; channel interpretation and boundary precision require geoscientist review.
- **Image coordinates are not geospatial coordinates:** rotated/resized image exports require careful transform handling before linking polygons to inline/crossline.
- **PHIT prediction is experimental:** validate on held-out wells and guard against well/sample leakage, repeated spatial samples, and time/depth alignment errors.
- **Sparse SEG-Y predictions:** interpolated time samples are not the same as measured or predicted full-volume coverage.
- **Gemma responses can be uncertain or invalid JSON:** the app validates report structure, but geological truth still requires external evidence.
- **Local infrastructure dependency:** the Gemma endpoint, dataset paths, and model files must be available on the APGCE environment.

## Next steps

- Wire the spectral decomposition, PHIT inference, and Mu-Rho buttons to validated batch jobs with progress reporting.
- Improve automatic polygon tracing using original spectral arrays and valid-data masks; georeference polygons to survey coordinates.
- Evaluate PHIT using blind-well splits and report realistic MAE/RMSE/R² and uncertainty.
- Perform reservoir ranking sensitivity checks, calibrate weights to well evidence, and avoid claiming hydrocarbon probabilities.
- Add reproducible configs, lightweight tests, and dependency pinning for deployment.

---

**Project principle:** the Copilot should help a geoscientist explain *why* a feature or candidate is interesting, *what alternative explanations exist*, and *what evidence is still needed*—rather than treating AI output as a final subsurface interpretation.
