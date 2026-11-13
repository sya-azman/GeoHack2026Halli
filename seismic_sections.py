
"""Plot reservoir candidates on co-registered SEG-Y inline attribute sections."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('/data/epic20/APGCE/input_data/challenge1_challenge2')
VOLUMES = {
    # 'Full Stack': ROOT / 'stack/Challenge1_Challenge2_Full_Stack_cropped.sgy',
    'VpVs': ROOT / 'inversion/Challenge1_Challenge2_Inversion_VpVs_cropped.sgy',
    'TotalSandProb': ROOT / 'inversion/Challenge1_Challenge2_Inversion_TotalSandProbability_cropped.sgy',
    'PHIT_pred': Path('outputs/PHIT_pred2.sgy'),
}
DEFAULT_CMAPS = {'VpVs': 'viridis', 'TotalSandProb': 'YlOrRd', 'PHIT_pred': 'plasma'}
COLORMAPS = ('gray', 'seismic', 'RdBu_r', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'YlOrRd', 'coolwarm')


@lru_cache(maxsize=4)
def _trace_headers(path_string):
    import segyio
    with segyio.open(path_string, 'r', ignore_geometry=True) as f:
        il = np.asarray(f.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32)
        xl = np.asarray(f.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32)
    return il, xl


def plot_attribute_section(candidate, volume='VpVs', cmap=None, half_width=60, half_time=220, max_traces=240, compact=False):
    import segyio
    if volume not in VOLUMES:
        raise ValueError(f'Unknown seismic volume: {volume}')
    path = VOLUMES[volume]
    if not path.is_file():
        raise FileNotFoundError(f'SEG-Y not found: {path}')
    il_target = int(candidate['Inline'])
    xl_target = int(candidate['Crossline'])
    twt_target = float(candidate['TWT'])
    il, xl = _trace_headers(str(path.resolve()))
    il_available = np.unique(il)
    nearest_il = int(il_available[np.argmin(np.abs(il_available.astype(np.int64) - il_target))])
    indices = np.flatnonzero((il == nearest_il) & (np.abs(xl.astype(np.int64) - xl_target) <= half_width))
    if len(indices) < 2:
        raise ValueError(f'Not enough traces near crossline {xl_target} on inline {nearest_il}')
    indices = indices[np.argsort(xl[indices])]
    if len(indices) > max_traces:
        indices = indices[np.linspace(0, len(indices) - 1, max_traces, dtype=int)]
    with segyio.open(str(path), 'r', ignore_geometry=True) as f:
        times = np.asarray(f.samples, dtype=float)
        keep = (times >= twt_target - half_time) & (times <= twt_target + half_time)
        if not keep.any():
            raise ValueError(f'TWT {twt_target:g} ms outside {volume} sample range')
        section = np.stack([np.asarray(f.trace[int(i)], dtype=np.float32)[keep] for i in indices], axis=1)
        times = times[keep]
    section[~np.isfinite(section)] = np.nan
    # Inversion volumes often use zero outside the modelled interval.
    if volume != 'Full Stack':
        section[section == 0] = np.nan
    if not np.isfinite(section).any():
        raise ValueError(f'No valid {volume} samples in this window')
    valid = section[np.isfinite(section)]
    if volume == 'Full Stack':
        clip = float(np.percentile(np.abs(valid), 98))
        vmin, vmax = -clip, clip
    elif volume == 'TotalSandProb':
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(valid, [2, 98])
    if not np.isfinite(vmin + vmax) or vmin >= vmax:
        vmin, vmax = float(np.min(valid)) - 1e-6, float(np.max(valid)) + 1e-6
    fig, ax = plt.subplots(figsize=(4.7, 4.2) if compact else (9, 4.5), layout='constrained')
    im = ax.imshow(section, aspect='auto', interpolation='nearest',
                   cmap=cmap or DEFAULT_CMAPS[volume], vmin=vmin, vmax=vmax,
                   extent=[float(xl[indices[0]]), float(xl[indices[-1]]), float(times[-1]), float(times[0])])
    ax.scatter([xl_target], [twt_target], s=70, c='#ffffff', edgecolors='black', linewidths=0.7, zorder=5)
    ax.annotate(f"Rank {int(candidate['rank'])}", (xl_target, twt_target), xytext=(5, -10),
                textcoords='offset points', color='#ffdf40', weight='bold',
                bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none', 'pad': 2})
    ax.set(xlabel='Crossline', ylabel='TWT (ms)',
           title=f'{volume} | Inline {nearest_il} | Rank {int(candidate["rank"])}')
    ax.set_ylim(float(times[-1]), float(times[0]))
    fig.colorbar(im, ax=ax, label=volume, shrink=.75, fraction=.055, pad=.02)
    if compact:
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
    return fig
