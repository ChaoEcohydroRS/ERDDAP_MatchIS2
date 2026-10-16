"""figkit — publication-figure conventions for geospatial / remote-sensing papers.

A thin matplotlib helper that bakes in the palette, fonts, colorbar layouts,
inference-cache I/O, and PNG+PDF saving developed across paper figures. Import it
so every figure starts from the same proven defaults instead of re-deriving them.

Pure matplotlib + numpy on purpose — no cartopy / earthengine dependency, so
`import figkit` works in any plotting env. The cartopy + contextily + GEE
study-area recipes live in `cartopy-recipes.md` (copy-paste, not importable).

    import figkit as fk          # vendored copy; lives next to fig_is2_validation.py
    fk.set_pub_style()
    fig, ax = plt.subplots()
    ...
    fk.panel_label(ax, 'a', 'Ablation')
    fk.save_fig(fig, "figures/fig3_ablation.png")   # writes .png AND .pdf at 300 dpi

Gallery / self-test (exercises most helpers, doubles as a smoke test):
    uv run --with matplotlib --with numpy python figkit.py demo /tmp/figkit_demo.png
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Color palette  (cross-figure consistency — same colors mean the same thing)
# ---------------------------------------------------------------------------
# Generalize the SAR family / fusion names to YOUR model lineup as needed; keep
# the *roles* (optical=green, the radar family = light→dark blue ramp, the
# fusion/combined model = red) consistent across every figure in a paper.

OPTICAL = '#27ae60'                                    # optical / NAIP / RGB family
SAR = ['#85c1e9', '#5dade2', '#2e86c1', '#1a5276']     # radar family, light → dark
FUSION = '#e74c3c'                                     # fusion / combined model
FUSION_DARK = '#922b21'                                # second fusion variant

PALETTE = {'optical': OPTICAL, 'sar': SAR,
           'fusion': FUSION, 'fusion_dark': FUSION_DARK}

# Named single-purpose colormaps (strings = matplotlib built-ins).
ERROR_MAGNITUDE_CMAP = 'magma_r'   # |error| maps: white → orange → magenta → black
SIGNED_DIFF_CMAP = 'RdBu_r'        # signed error / A-minus-B difference maps


def cmap_contribution():
    """Green↔blue diverging map for an A-vs-B fraction (e.g. optical-vs-SAR share).

    Green end = source A dominates, blue end = source B dominates, white = even.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'optical_sar',
        ['#27ae60', '#a9dfbf', '#ffffff', '#aed6f1', '#2e86c1'], N=256)


def cmap_river():
    """Blue ramp for river stream-order (low order = big mainstem = dark blue)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'riv_ord',
        ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff'], N=256)


def signed_norm(vmax, vcenter=0.0, vmin=None):
    """TwoSlopeNorm pinned at `vcenter` for signed difference maps.

    Use with SIGNED_DIFF_CMAP so zero maps to white regardless of asymmetry:
        im = ax.imshow(diff, cmap=fk.SIGNED_DIFF_CMAP, norm=fk.signed_norm(3.0))
    """
    from matplotlib.colors import TwoSlopeNorm
    if vmin is None:
        vmin = -vmax
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


# ---------------------------------------------------------------------------
# Global style  (fonts / sizes that read well at print size)
# ---------------------------------------------------------------------------
def set_pub_style():
    """Set matplotlib rcParams to the paper defaults. Call once at script start.

    Panel titles 13pt bold left-aligned; axis labels 12pt; ticks/legend ~10pt;
    no top/right spines; 300-dpi tight-bbox saves. Nothing here is below 10pt —
    sub-10pt text looks unprofessional at print size.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',   # panel labels are left-aligned
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ---------------------------------------------------------------------------
# Panel labels  (a)/(b)/(c) — bold, left-aligned, optional non-bold annotation
# ---------------------------------------------------------------------------
def panel_label(ax, letter, bold=None, plain=None, fontsize=13, loc='left', pad=None):
    """Left-aligned bold panel label, optionally with a trailing non-bold note.

        panel_label(ax, 'a')                  -> "(a)"                  (all bold)
        panel_label(ax, 'a', 'Ablation')      -> "(a) Ablation"         (all bold)
        panel_label(ax, 'a', 'Patch 164', plain='mean=5.4m')
            -> "(a) Patch 164" bold  +  "  mean=5.4m" non-bold   (via mathtext)

    The bold/plain split uses mathtext, so keep `bold` to plain words/numbers
    (no % $ _ ^ unless you escape them).
    """
    if plain:
        prefix = f'({letter})' + (f' {bold}' if bold else '')
        bold_mt = prefix.replace(' ', r'\ ')
        ax.set_title(rf'$\bf{{{bold_mt}}}$  {plain}', fontsize=fontsize,
                     loc=loc, pad=pad)
    else:
        label = f'({letter})' + (f' {bold}' if bold else '')
        ax.set_title(label, fontsize=fontsize, fontweight='bold', loc=loc, pad=pad)


# ---------------------------------------------------------------------------
# Bar-top value labels  (with automatic headroom so labels clear the title)
# ---------------------------------------------------------------------------
def bar_value_labels(ax, bars, values, fmt='{:.3f}', headroom=0.15, fontsize=10):
    """Annotate each bar with its value and extend the y-axis so labels fit.

        bars = ax.bar(x, values, color=[fk.OPTICAL, *fk.SAR])
        fk.bar_value_labels(ax, bars, values)
    """
    vmax = max(values)
    ax.set_ylim(0, vmax * (1 + headroom))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vmax * 0.02,
                fmt.format(val), ha='center', va='bottom', fontsize=fontsize)


# ---------------------------------------------------------------------------
# Colorbars  (shrink to the middle `frac` of the cell — less cramped)
# ---------------------------------------------------------------------------
def shrink_cbar(cax, frac=0.8, orientation='vertical'):
    """Shrink a colorbar axis to the middle `frac` of its gridspec cell."""
    pos = cax.get_position()
    pad = (1 - frac) / 2
    if orientation == 'vertical':
        cax.set_position([pos.x0, pos.y0 + pos.height * pad,
                          pos.width, pos.height * frac])
    else:
        cax.set_position([pos.x0 + pos.width * pad, pos.y0,
                          pos.width * frac, pos.height])


def image_row(fig, images, cmaps=None, norms=None, vmins=None, vmaxs=None,
              labels=None, panel_titles=None, wspace=0.06, cbar_frac=0.8):
    """Row of image panels, each with its OWN horizontal colorbar directly below.

    Best layout for multi-panel image comparisons — the reader never has to guess
    which colorbar matches which panel. Returns (axes, caxes, ims).

        axes, caxes, ims = fk.image_row(
            fig, [err_a, err_b], cmaps=[fk.ERROR_MAGNITUDE_CMAP]*2,
            labels=['|error| (m)']*2, panel_titles=['(a) Optical', '(b) Fusion'])
    """
    import matplotlib.gridspec as gridspec
    n = len(images)
    cmaps = cmaps or ['viridis'] * n
    gs = gridspec.GridSpec(2, n, height_ratios=[1, 0.045],
                           wspace=wspace, hspace=0.02, figure=fig)
    axes, caxes, ims = [], [], []
    for col in range(n):
        ax = fig.add_subplot(gs[0, col])
        ax.set_anchor('S')           # anchor image to bottom so cbar sits flush
        kw = {}
        if norms and norms[col] is not None:
            kw['norm'] = norms[col]
        else:
            if vmins:
                kw['vmin'] = vmins[col]
            if vmaxs:
                kw['vmax'] = vmaxs[col]
        im = ax.imshow(images[col], cmap=cmaps[col], **kw)
        ax.axis('off')
        if panel_titles:
            ax.set_title(panel_titles[col], fontsize=13, fontweight='bold', loc='left')
        cax = fig.add_subplot(gs[1, col])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        if labels:
            cb.set_label(labels[col], fontsize=11)
        shrink_cbar(cax, frac=cbar_frac, orientation='horizontal')
        axes.append(ax); caxes.append(cax); ims.append(im)
    return axes, caxes, ims


# ---------------------------------------------------------------------------
# Saving  (always PNG + PDF at 300 dpi for paper-going figures)
# ---------------------------------------------------------------------------
def save_fig(fig, path, dpi=300, pdf=True):
    """Save a figure as PNG and (by default) a vector PDF, creating the dir.

    PDF is editable in Illustrator/Inkscape for late-stage tweaks.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if pdf:
        fig.savefig(os.path.splitext(path)[0] + '.pdf', bbox_inches='tight')
    return path


# ---------------------------------------------------------------------------
# Inference caching  (cache GPU/heavy compute so plotting tweaks run in seconds)
# ---------------------------------------------------------------------------
def save_cache(path, data):
    """Save a dict of arrays/scalars to a compressed .npz (scalars become 0-d)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {k: (v if isinstance(v, np.ndarray) else np.array(v))
               for k, v in data.items()}
    np.savez_compressed(path, **payload)


def load_cache(path):
    """Load a .npz written by save_cache back into a plain dict.

    Scalars come back as 0-d arrays — call `.item()` (e.g. shape tuples stored as
    arrays: `tuple(d['mosaic_shape'])`).
    """
    c = np.load(path, allow_pickle=True)
    return {k: c[k] for k in c.files}


def cached(path, compute, force=False):
    """Return cached dict if `path` exists (and not `force`), else compute+cache.

        data = fk.cached('results/cache_fig8.npz',
                         lambda: run_inference(args), force=args.force)

    The cache key (filename) MUST encode every arg that changes predictions
    (model dir, tile list, threshold). Change one of those → pass force=True.
    """
    if os.path.exists(path) and not force:
        return load_cache(path)
    data = compute()
    save_cache(path, data)
    return data


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
def _demo(out_path):
    """Render a 1×3 gallery exercising the helpers; doubles as a smoke test."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    set_pub_style()
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(14, 4.2))
    outer = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.22,
                              figure=fig)

    # (a) bar chart with palette + value labels + headroom
    ax_a = fig.add_subplot(outer[0])
    names = ['Optical', 'SAR-a', 'SAR-b', 'SAR-c', 'Fusion']
    vals = [0.62, 0.71, 0.74, 0.78, 0.83]
    colors = [OPTICAL, SAR[0], SAR[1], SAR[2], FUSION]
    bars = ax_a.bar(names, vals, color=colors)
    bar_value_labels(ax_a, bars, vals, fmt='{:.2f}')
    ax_a.set_ylabel(r'R$^2$')
    panel_label(ax_a, 'a', 'Ablation')

    # (b) signed difference map with diverging norm + shrunk vertical cbar
    sub = outer[1].subgridspec(1, 2, width_ratios=[1, 0.045], wspace=0.05)
    ax_b = fig.add_subplot(sub[0])
    diff = rng.normal(0, 1.2, (60, 60)).cumsum(0).cumsum(1)
    diff = diff / np.abs(diff).max() * 3
    im_b = ax_b.imshow(diff, cmap=SIGNED_DIFF_CMAP, norm=signed_norm(3.0))
    ax_b.axis('off')
    panel_label(ax_b, 'b', 'A − B (m)')
    cax_b = fig.add_subplot(sub[1])
    fig.colorbar(im_b, cax=cax_b)
    shrink_cbar(cax_b, frac=0.8, orientation='vertical')

    # (c) contribution fraction map with custom diverging cmap + horizontal cbar
    sub_c = outer[2].subgridspec(2, 1, height_ratios=[1, 0.05], hspace=0.04)
    ax_c = fig.add_subplot(sub_c[0])
    ax_c.set_anchor('S')
    frac = rng.uniform(-1, 1, (60, 60))
    frac = np.clip(frac + np.linspace(-0.5, 0.5, 60)[None, :], -1, 1)
    im_c = ax_c.imshow(frac, cmap=cmap_contribution(), vmin=-1, vmax=1)
    ax_c.axis('off')
    panel_label(ax_c, 'c', 'SAR fraction')
    cax_c = fig.add_subplot(sub_c[1])
    cb_c = fig.colorbar(im_c, cax=cax_c, orientation='horizontal')
    cb_c.set_label('← optical    SAR →', fontsize=11)
    shrink_cbar(cax_c, frac=0.8, orientation='horizontal')

    save_fig(fig, out_path)
    plt.close(fig)

    # exercise the cache round-trip too
    tmp_cache = os.path.join(os.path.dirname(out_path) or '.', '_figkit_cache_test.npz')
    save_cache(tmp_cache, {'pred': frac, 'mosaic_shape': frac.shape, 'rmse': 1.23})
    back = load_cache(tmp_cache)
    assert tuple(back['mosaic_shape']) == frac.shape
    assert abs(float(back['rmse']) - 1.23) < 1e-6
    os.remove(tmp_cache)

    print(f'wrote {out_path} and {os.path.splitext(out_path)[0]}.pdf')
    print('cache round-trip OK')


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == 'demo':
        out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/figkit_demo.png'
        _demo(out)
    else:
        print(__doc__)
        print('Usage: python figkit.py demo [out.png]')
