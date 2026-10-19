"""Export one plot as a 600-dpi PNG and a vector PDF, with panel labels."""
from pathlib import Path
from string import ascii_uppercase


def add_panel_labels(figure, axes=None):
    """Label plotting axes in their creation order; do not label colour bars."""
    panels = list(axes) if axes is not None else [
        ax for ax in figure.axes
        if ax.get_visible() and ax.get_label() != '<colorbar>'
        and not hasattr(ax, '_colorbar')
    ]
    if len(panels) < 2:
        return []
    if len(panels) > len(ascii_uppercase):
        raise ValueError('Split figures with more than 26 panels before exporting')
    labels = []
    for letter, ax in zip(ascii_uppercase, panels):
        existing = getattr(ax, '_hb_panel_label', None)
        if existing is not None:
            existing.set_text(letter)
            labels.append(existing)
            continue
        draw_text = ax.text2D if ax.name == '3d' else ax.text
        label = draw_text(
            .015, .985, letter, transform=ax.transAxes,
            ha='left', va='top', fontsize=14, fontweight='bold',
            color='black', zorder=1000, clip_on=False,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5),
        )
        label.set_in_layout(False)
        ax._hb_panel_label = label
        labels.append(label)
    return labels


def save_figure(figure, path, *, dpi=600, **kwargs):
    """Keep each existing PNG name and also save the same figure as a PDF.

    Both outputs use the original Matplotlib artists. No raster conversion,
    resampling of the data, or modification of plotted values is performed.
    """
    import matplotlib

    path = Path(path)
    if path.suffix.lower() not in {'.png', '.pdf'}:
        raise ValueError('Figure output must have a .png or .pdf suffix')
    path.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = path.with_suffix('.png'), path.with_suffix('.pdf')
    png_dpi = max(600, int(dpi))
    add_panel_labels(figure)
    kwargs.setdefault('bbox_inches', 'tight')
    with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42,
                                'pdf.compression': 9}):
        figure.savefig(png, dpi=png_dpi, **kwargs)
        figure.savefig(pdf, **kwargs)
    return png, pdf
