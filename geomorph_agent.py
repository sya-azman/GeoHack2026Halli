"""Dr. Geomorph: original prompts and measurements, with APGCE Gemma inference."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path
import cv2
import numpy as np
from apgce_client import MODEL, get_client

PROJECT_INFO = {
    "formation": "Oligocene",
    "frequencies_hz": {"red": 15, "green": 30, "blue": 60},
    "attribute": "RGB Spectral Decomposition",
    "twt_ms": 1700,
}

SYSTEM = """You are Dr. Geomorph, a senior geoscientist with 30 years of
experience interpreting seismic geomorphology and spectral decomposition.

- Use the specified RGB frequencies, or state an assumed low/mid/high mapping.
  Low-frequency dominance may indicate thicker or more attenuating bodies and
  high-frequency dominance may indicate thinner beds. Consider tuning and
  display scaling. Colour alone cannot establish thickness or lithology.
  
- Describe geometry first, then spectral response, then name the feature.
  Discuss sinuosity, width, bifurcations, lobe shape and truncations where visible.
  Flag acquisition and processing artefacts. Contours are image measurements
  that can include plot borders and labels.
- Spectral DHIs are indirect. Give false-positive risks and validation needs
  for each candidate. Use an empty list if none are credible.
  
- Confidence >=0.8 requires unambiguous multi-criteria evidence. Scores from
  0.5 to below 0.8 indicate plausible interpretations with alternatives.
  Scores below 0.5 indicate speculation. State what remains unresolved.
  
- Use one polygon per body, with 4-20 integer [x,y] vertices on a 0-1000 grid
  relative to the full named image, including margins. Origin is top left.
  Trace the visible boundary.
  

- Never label the entire survey or a broad spectral patch as a a polygon.

- Avoid annotations, well markers, plot titles and black no-data regions.

- A polygon is an approximate localization, NOT a precise geological boundary.
  Return features=[] if no convincing bounded features are visible.
- Never use visual brightness alone as evidence of hydrocarbons.
Output only one JSON object, without Markdown or commentary."""

TEMPLATE = """{
  "summary": "3-6 sentences",
  "features": [{"id":"F1","image":"<label>","name":"...",
    "category":"channel|point bar|crevasse splay|levee|distributary|mouth bar|delta lobe|incised valley|strandplain|tidal channel|slope channel|fan lobe|MTD|carbonate buildup|fault|karst|artefact|other",
    "description":"geometry, extent, orientation","spectral_evidence":"...",
    "polygon":[[0,0],[1,0],[1,1],[0,1]],"confidence":0.0,"why":"..."}],
  "environment":{"primary":"...",
    "system":"fluvial|deltaic|shoreface|shelf|slope|basin floor|carbonate|mixed",
    "alternatives":["..."],"reasoning":"...","supporting_features":["F1"],"confidence":0.0},
  "dhis":[{"id":"D1","image":"<label>","type":"...","description":"...",
    "spectral_evidence":"...","polygon":[[0,0],[1,0],[1,1],[0,1]],
    "confidence":0.0,"false_positive_risks":["..."],"validation_needed":["..."]}],
  "overall_confidence":{"score":0.0,"level":"high|moderate|low|very_low",
    "limiting_factors":["..."],"additional_data":["..."]}
}"""

def measure_image(path):
    import cv2
 
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        _, _, width, height = cv2.boundingRect(contour)
        features.append({
            "area_px2": round(area, 2),
            "perimeter_px": round(cv2.arcLength(contour, True), 2),
            "width_px": width,
            "height_px": height,
            "aspect_ratio": round(width / height, 2),
        })
    blue, green, red = np.mean(img, axis=(0, 1))
    return {
        "number_of_contours": len(features),
        "largest_contours": sorted(features, key=lambda item: item["area_px2"], reverse=True)[:20],
        "full_image_rgb_means": {
            "red": round(float(red), 2),
            "green": round(float(green), 2),
            "blue": round(float(blue), 2),
        },
    }

def auto_trace_polygons(
    image_path,
    percentile=92,
    top_n=8,
    min_area=110,
    min_elongation=2.2
):
    """Detect bright elongated features using backend grayscale processing.

    Returns polygons in normalized 0-1000 image coordinates.
    Does not modify or save the original RGB image.
    """
    from scipy import ndimage as ndi

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape

    # Black pixels are treated as no-data.
    # Best used on clean spectral slices without annotations.
    valid = gray > 12

    # Avoid detecting plot edges
    distance = ndi.distance_transform_edt(valid)
    valid &= distance > 10

    if not valid.any():
        return []

    # Multiscale bright ridge detection
    response = np.zeros_like(gray)

    for scale in (2.0, 3.5, 5.0):
        dxx = ndi.gaussian_filter(gray, scale, order=(0, 2)) * scale**2
        dyy = ndi.gaussian_filter(gray, scale, order=(2, 0)) * scale**2
        dxy = ndi.gaussian_filter(gray, scale, order=(1, 1)) * scale**2

        delta = np.hypot(dxx - dyy, 2 * dxy)

        eig_small = (dxx + dyy - delta) / 2
        eig_large = (dxx + dyy + delta) / 2

        ridge_response = -eig_small - np.abs(eig_large)
        response = np.maximum(response, ridge_response)

    threshold = np.percentile(response[valid], percentile)

    ridge = (response >= threshold) & valid
    ridge = ndi.binary_closing(ridge, iterations=3) & valid

    labels, count = ndi.label(
        ridge,
        structure=np.ones((3, 3))
    )

    candidates = []

    for index in range(1, count + 1):
        mask = labels == index

        if mask.sum() < min_area:
            continue

        yy, xx = np.where(mask)

        if len(xx) < 3:
            continue

        covariance = np.cov(xx, yy)
        eigenvalues = np.linalg.eigvalsh(covariance)

        minor, major = eigenvalues

        elongation = np.sqrt(major / max(minor, 1.0))

        if major < 85 or elongation < min_elongation:
            continue

        # Extract actual connected-component outline
        contour_mask = (ndi.binary_fill_holes(mask).astype(np.uint8) * 255)

        contours, _ = cv2.findContours(contour_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(contour, True)

        # Simplify the polygon
        approx = cv2.approxPolyDP(contour,epsilon=0.01 * perimeter,closed=True)

        # Simplify further if necessary
        epsilon = 0.01
        while len(approx) > 20 and epsilon < 0.15:
            epsilon += 0.01
            approx = cv2.approxPolyDP(contour,epsilon * perimeter,True)

        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue

        # Convert pixels to Gemma's normalized coordinates
        polygon = [[int(np.clip(round(x / max(width - 1, 1) * 1000), 0, 1000)),
                int(np.clip(round(y / max(height - 1, 1) * 1000), 0, 1000))]
            for x, y in points]

        score = float(np.sqrt(major) * np.sqrt(elongation))

        candidates.append({
            "polygon": polygon,
            "geometry_score": round(score, 3),
            "elongation": round(float(elongation), 2),
            "area_px": int(mask.sum()),
            "centroid": [
                float(np.mean(xx) / max(width - 1, 1) * 1000),
                float(np.mean(yy) / max(height - 1, 1) * 1000)]})

    candidates.sort(key=lambda item: item["geometry_score"],reverse=True)

    return candidates[:top_n]

def merge_auto_polygons(report, auto_polygons):
    """Refine Gemma feature polygons using overlapping automatic detections."""

    updated = dict(report)
    updated_features = []
    used = set()

    for feature in report.get("features", []):
        item = dict(feature)

        gemma_polygon = np.asarray(item.get("polygon", []),dtype=np.float32)

        if gemma_polygon.ndim != 2 or gemma_polygon.shape[0] < 3:
            updated_features.append(item)
            continue

        if gemma_polygon.shape[1] != 2:
            updated_features.append(item)
            continue

        if not np.isfinite(gemma_polygon).all():
            updated_features.append(item)
            continue

        # Compare with automatically detected polygon centroids
        matches = []

        for i, candidate in enumerate(auto_polygons):
            if i in used:
                continue

            center = tuple(candidate["centroid"])
            inside = cv2.pointPolygonTest(gemma_polygon,center,False)

            if inside >= 0:
                matches.append((i, candidate))

        if matches:
            # Choose strongest matching ridge
            best_i, best = max(
                matches,
                key=lambda x: x[1]["geometry_score"])

            item["polygon"] = best["polygon"]
            item["why"] = (
                str(item.get("why", "")) +" [Polygon refined by grayscale ridge detection; boundary is approximate.]")

            used.add(best_i)

        updated_features.append(item)

    updated["features"] = updated_features

    return updated
    
def extract_json(text):
    if not text:
        raise ValueError("Model returned an empty response")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("No JSON object found in model response")
        return json.loads(text[start:end+1])


def interpret_seismic(image_path, client=None, model=MODEL, project_info=None):
    image_path = Path(image_path)
    client = client if client is not None else get_client()
    project_info = project_info if project_info is not None else PROJECT_INFO
    from PIL import Image
    with Image.open(image_path) as im:
        measurements = {"image_width": im.width, "image_height": im.height}
    prompt = f"""Analyze the attached image labelled {image_path.name}.
Project information: {json.dumps(project_info)}
Image dimensions: {measurements.get("image_width", "unknown")} x {measurements.get("image_height", "unknown")}.
Ignore plot annotations and black margins. Do not treat Canny contours as geological evidence.
Use this exact report structure. Pipe-delimited values are choices.
{TEMPLATE}
Discuss reservoir distribution, connectivity, possible sand fairways, quality,
seal and charge where supportable in the summary or limiting factors. If
warranted, order potential targets by evidence in the summary and explain why.
Distinguish geomorphology confidence from confidence in reservoir presence.
Use the actual image filename in all image fields.
Return a single JSON object only."""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Only PNG, JPEG and WEBP images are supported")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ]},
        ],
        temperature=0.2,
    )
    report = extract_json(response.choices[0].message.content)
    expected_keys = set(json.loads(TEMPLATE))
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError(f"Unexpected report keys: {list(report) if isinstance(report,dict) else type(report)}; expected {sorted(expected_keys)}")
    if not isinstance(report.get("features"), list):
        raise ValueError("Report features must be a list")

    # Backend grayscale polygon tracing
    auto_polygons = auto_trace_polygons(image_path,percentile=92,top_n=8)

    # Refine Gemma polygons where detections overlap
    report = merge_auto_polygons(report,auto_polygons)
    
    return report


def dr_geomorph(argv=None, client=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=Path("spectral_interpretation_report.json"))
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args(argv)
    if args.measure_only:
        print(json.dumps(measure_image(args.image), indent=2))
        return
    report = interpret_seismic(args.image, client=client, model=args.model)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    dr_geomorph()


def filter_unreliable_polygons(report, image_size, max_area_fraction=0.18):
    """Suppress implausibly broad polygons; does not validate geological interpretation.

    Removed features remain in report with polygon=[] and a caution in `why`.
    """
    import cv2
    import numpy as np
    cleaned = dict(report)
    features = []
    for feature in report.get('features', []):
        item = dict(feature)
        poly = item.get('polygon', [])
        try:
            pts = np.asarray(poly, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
                raise ValueError('invalid vertices')
            if (pts < 0).any() or (pts > 1000).any():
                raise ValueError('outside image')
            area_fraction = abs(cv2.contourArea(pts.astype(np.float32))) / 1_000_000
            if area_fraction > max_area_fraction:
                item['polygon'] = []
                item['why'] = (str(item.get('why', '')) +
                    f' [Polygon withheld: covers {area_fraction:.0%} of image; manual review required.]')
        except (ValueError, TypeError):
            item['polygon'] = []
        features.append(item)
    cleaned['features'] = features
    return cleaned
