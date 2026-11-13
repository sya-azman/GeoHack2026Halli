"""Streamlit MVP: geobody detection, reservoir screening and exploration chat."""
import io
import hashlib
import json
import re
from tempfile import TemporaryDirectory
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from apgce_client import MODEL, BASE_URL, get_client
from geomorph_agent import PROJECT_INFO, interpret_seismic, measure_image, filter_unreliable_polygons
from reservoir_screening import screen_reservoirs
from seismic_sections import (plot_attribute_section, COLORMAPS, DEFAULT_CMAPS)

def draw_gemma_overlay(image, features):
    """Draw full-image normalized (0..1000) Gemma polygons; no spatial calibration."""
    result = image.convert('RGB').copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 32)
    except OSError:
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 32)
        except OSError:
            font = ImageFont.load_default()
    w, h = result.size
    palette = ['#ffe64a', '#54f6ff', '#91ff82', '#ff7dfa', '#ff8364', '#b685ff']
    for idx, feature in enumerate(features):
        polygon = feature.get('polygon', [])
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        try:
            pts = []
            for point in polygon:
                x, y = map(float, point)
                if not (0 <= x <= 1000 and 0 <= y <= 1000):
                    raise ValueError('Polygon coordinate outside 0..1000')
                pts.append((round(x * (w - 1) / 1000), round(y * (h - 1) / 1000)))
        except (ValueError, TypeError):
            continue
        if len(pts) < 3:
            continue
        color = palette[idx % len(palette)]
        draw.line(pts + [pts[0]], fill=color, width=3, joint='curve')
        label = str(feature.get('id') or f'F{idx+1}')
        bbox = draw.textbbox((0, 0), label, font=font, stroke_width=2)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx = max(3, min(pts[0][0] + 8, w - tw - 16))
        ly = max(3, min(pts[0][1] - th - 14, h - th - 16))
        draw.rounded_rectangle((lx - 5, ly - 5, lx + tw + 9, ly + th + 9), radius=5, fill='#101010')
        draw.text((lx + 2 - bbox[0], ly + 2 - bbox[1]), label, font=font, fill=color)
    return result


def show_fitted_image(image, caption=None):
    """Fit preview within 850x560 px without modifying analysis data."""
    if isinstance(image, (bytes, bytearray)):
        source = Image.open(io.BytesIO(image)).convert('RGB')
    else:
        source = image.convert('RGB')
    preview = source.copy()
    preview.thumbnail((850, 560), Image.Resampling.LANCZOS)
    st.image(preview, caption=caption)


st.set_page_config(page_title='Dr Geomorph Seismic Exploration', layout='wide')
st.title('Seismic Exploration Copilot')
st.caption(f'Shared Gemma model: {MODEL} | {BASE_URL}')

uploaded = st.file_uploader('Upload RGB spectral decomposition image', type=['png', 'jpg', 'jpeg', 'webp'])
if uploaded is None:
    st.info('Upload an RGB spectral decomposition image to begin.')
raw = uploaded.getvalue() if uploaded is not None else None
image = Image.open(io.BytesIO(raw)).convert('RGB') if raw is not None else None
analysis_image = image
match = re.search(r'(\d+(?:\.\d+)?)ms(?=\.|_|$)', uploaded.name, re.I) if uploaded else None
filename_twt = float(match.group(1)) if match else None

with st.sidebar:
    st.header('Project settings')
    # formation = st.text_input('Formation', PROJECT_INFO['formation'])
    if filename_twt is not None:
        st.success(f'TWT detected from filename: {filename_twt:g} ms')
        twt = filename_twt
    else:
        if uploaded is not None:
            st.warning('No TWT in filename; enter it manually.')
        twt = st.number_input('TWT (ms)', min_value=0.0, value=0.0, step=4.0)
    red = st.number_input('Red frequency (Hz)', min_value=0, value=int(PROJECT_INFO['frequencies_hz']['red']))
    green = st.number_input('Green frequency (Hz)', min_value=0, value=int(PROJECT_INFO['frequencies_hz']['green']))
    blue = st.number_input('Blue frequency (Hz)', min_value=0, value=int(PROJECT_INFO['frequencies_hz']['blue']))

    st.divider()
    st.subheader('Seismic section display')
    selected_volumes = st.multiselect(
        'Volumes to display', ['VpVs', 'TotalSandProb', 'PHIT_pred'],
        default=['VpVs', 'TotalSandProb', 'PHIT_pred'])
    volume_cmaps = {}
    for vol in selected_volumes:
        volume_cmaps[vol] = st.selectbox(
            f'{vol} colour map', COLORMAPS,
            index=COLORMAPS.index(DEFAULT_CMAPS[vol]), key=f'cmap_{vol}')
        # Native Streamlit selectboxes cannot render gradient thumbnails inside options.
        # Display a small real colour-scale sample immediately below each selector.
        import numpy as np
        import matplotlib.pyplot as plt
        swatch, swatch_ax = plt.subplots(figsize=(3.8, .35))
        swatch_ax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto',
                         cmap=volume_cmaps[vol], extent=[0, 1, 0, 1])
        swatch_ax.axis('off')
        swatch.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(swatch, use_container_width=True)
        plt.close(swatch)


project = {
    # 'formation': formation,
    'frequencies_hz': {'red': red, 'green': green, 'blue': blue},
    'attribute': 'RGB Spectral Decomposition', 'twt_ms': twt,
}
settings_key = json.dumps(project, sort_keys=True)
image_key = (hashlib.sha256(raw + analysis_image.tobytes() + settings_key.encode()).hexdigest()
             if image is not None else None)

st.header('1. Geobody detection — Dr. Geomorph')
run = False
if uploaded is not None:
    col1, col2 = st.columns([1.4, 1])
    with col1:
        show_fitted_image(image, caption=uploaded.name)
    with col2:
        st.write(f'Image: {image.width} × {image.height} pixels')
        if st.button('Measure image (no LLM)'):
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / ('input' + Path(uploaded.name).suffix.lower())
                path.write_bytes(raw)
                st.json(measure_image(path))
        run = st.button('Interpret with Gemma', type='primary')
else:
    st.info('Upload an RGB spectral decomposition image to run Dr. Geomorph.')

if run:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / ('input' + Path(uploaded.name).suffix.lower())
        analysis_image.save(path)
        try:
            with st.spinner('Running Dr. Geomorph on Gemma...'):
                report = interpret_seismic(path, client=get_client(), project_info=project)
            # Backend uses temporary input filename; normalize references to actual upload name.
            for item in report.get('features', []) + report.get('dhis', []):
                if isinstance(item, dict):
                    item['image'] = uploaded.name
            report = filter_unreliable_polygons(report, analysis_image.size)
            st.session_state['report'] = report
            st.session_state['report_key'] = image_key
            # st.session_state.pop('oil_finder_result', None)
        except Exception as exc:
            st.error(f'Interpretation failed: {exc}')


report = st.session_state.get('report')
if image is not None and report and st.session_state.get('report_key') == image_key:
    st.subheader('Dr. Geomorph interpretation')
    st.write(report.get('summary', ''))
    st.json(report.get('environment', {}), expanded=False)
    st.write('**Detected features**')
    st.dataframe(report.get('features', []), use_container_width=True)
    st.write('**Potential DHIs**')
    st.dataframe(report.get('dhis', []), use_container_width=True)
    st.write('**Overall confidence**')
    st.json(report.get('overall_confidence', {}), expanded=False)
    st.download_button('Download interpretation JSON', json.dumps(report, indent=2), 'spectral_interpretation_report.json', 'application/json')

    st.caption('Gemma polygons are unverified proposals, not validated channel boundaries.')
    overlay = draw_gemma_overlay(image, report.get('features', []))
    show_fitted_image(overlay, caption='AI feature polygon overlay (unverified proposals)')
    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    st.download_button('Download polygon overlay', buf.getvalue(), 'geomorph_overlay.png', 'image/png')


st.divider()
st.header('2. Reservoir screening')
uploaded_csv = st.file_uploader(
    "Upload seismic attributes CSV (max 200 MB)",
    type=["csv"],
    accept_multiple_files=False,
    key="attribute_csv"
)
    
if uploaded_csv is not None:
    if st.button('Run reservoir screening'):
        try:
            with st.spinner('Screening potential reservoir-quality samples...'):
                source_df = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()))
                candidates, summary = screen_reservoirs(source_df)
            st.session_state['reservoir_candidates'] = candidates
            st.session_state['reservoir_summary'] = summary
            st.session_state['reservoir_csv_key'] = hashlib.sha256(uploaded_csv.getvalue()).hexdigest()
        except Exception as exc:
            st.error(f'Reservoir screening failed: {exc}')
    if ('reservoir_summary' in st.session_state and
        st.session_state.get('reservoir_csv_key') == hashlib.sha256(uploaded_csv.getvalue()).hexdigest()):
        info = st.session_state['reservoir_summary']
        st.write(f"Valid samples: {info['valid_rows']:,} | Potential reservoir samples: {info['qualifying_rows']:,}")
        st.dataframe(st.session_state['reservoir_candidates'], use_container_width=True)
        st.download_button('Download reservoir candidates',
                           st.session_state['reservoir_candidates'].to_csv(index=False),
                           'reservoir_candidates.csv', 'text/csv')
        st.caption('Screening uses sampled CSV rows only; these are not mapped connected bodies or proven hydrocarbons.')
        st.subheader('Top 3 candidates on seismic sections')
        st.caption('Real SEG-Y sections with ranked sample markers. Inversion zeros are treated as no-data.')
        for _, candidate in st.session_state['reservoir_candidates'].head(3).iterrows():
            st.markdown(f"**Rank {int(candidate['rank'])} — IL {int(candidate['Inline'])}, XL {int(candidate['Crossline'])}, TWT {float(candidate['TWT']):g} ms**")
            if selected_volumes:
                columns = st.columns(len(selected_volumes))
                for column, volume in zip(columns, selected_volumes):
                    with column:
                        try:
                            with st.spinner(f'Loading {volume}...'):
                                fig = plot_attribute_section(candidate, volume, cmap=volume_cmaps[volume], compact=True)
                            st.pyplot(fig, use_container_width=True)
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                        except Exception as exc:
                            st.warning(f'Could not display {volume}: {exc}')

# Chatbot can use the most recent reports, without re-running either analysis.
st.divider()
st.header('3. Exploration chatbot')
st.caption('Ask Gemma about the Dr. Geomorph interpretation and ranked reservoir samples.')

active_report = (st.session_state.get('report')
                 if image_key is not None and st.session_state.get('report_key') == image_key
                 else None)
ranked = (st.session_state.get('reservoir_candidates')
          if uploaded_csv is not None and st.session_state.get('reservoir_csv_key') == hashlib.sha256(uploaded_csv.getvalue()).hexdigest()
          else None)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if st.button('Clear chat'):
    st.session_state['chat_history'] = []

for message in st.session_state['chat_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

question = st.chat_input('Ask about geomorphology, reservoir ranking, or interpretation risks...')
if question:
    st.session_state['chat_history'].append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    context = {
        'project': project if uploaded is not None else None,
        'geomorph_report': active_report,
        'reservoir_summary': (st.session_state.get('reservoir_summary')
                              if ranked is not None else None),
        'top_reservoir_candidates': (ranked.head(10).to_dict(orient='records')
                                     if ranked is not None else []),
    }
    system_prompt = (
        'You are an exploration geoscience copilot. Answer questions using the '
        'supplied interpretation report and reservoir screening results when relevant. '
        'Clearly distinguish observed evidence, model predictions, and hypotheses. '
        'If the context lacks evidence, say what is missing. '
        'Do not invent well calibration, traps, seals, or fluid evidence.\n'
        'Current analysis context (JSON):\n' + json.dumps(context, default=str)
    )
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(st.session_state['chat_history'][-12:])
    try:
        with st.spinner('Asking Gemma...'):
            response = get_client().chat.completions.create(
                model=MODEL, messages=messages, temperature=0.2
            )
            answer = response.choices[0].message.content or 'No response returned.'
        st.session_state['chat_history'].append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
    except Exception as exc:
        st.error(f'Chat request failed: {exc}')
        st.session_state['chat_history'].pop()  # Allow retry without duplicate question.

st.divider()
st.header('4. Processing workflows (placeholder, accepting seismic cube input in .segy format)')
st.caption('These are placeholders only')
workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
with workflow_col1:
    if st.button('Generate spectral decomposition', use_container_width=True):
        st.info('Spectral decomposition generation from segy volume')
with workflow_col2:
    if st.button('Generate PHIT prediction cube', use_container_width=True):
        st.info('Run PHIT prediction on segy using pre-trained PHIT model')
with workflow_col3:
    if st.button('Calculate Mu-Rho', use_container_width=True):
        st.info('Mu-Rho calculation from seismic P- and S-impedance volumes.')
        st.latex(r'\mu\rho = Z_S^2 = (V_S \rho)^2')
        st.caption(
            'Where Z_S is shear-wave impedance, '
            'V_S is shear-wave velocity, and ρ is density.')
