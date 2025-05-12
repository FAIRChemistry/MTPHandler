import itertools as it

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mtphandler.model import Plate


def visualize_plate(
    plate: Plate,
    name: str,
    zoom: bool = False,
    wavelengths: list[float] = [],
    static: bool = False,
    darkmode: bool = False,
):
    """Visualize a plate with all its wells and measurements."""

    if darkmode:
        theme = "plotly_dark"
        plot_bgcolor = "#1e1e1e"  # Dark background color for subplots
        paper_bgcolor = "#1e1e1e"
        gridcolor = plot_bgcolor  # Grid color for dark mode
        font_color = "#e5e5e5"  # Lighter text for dark mode
    else:
        theme = "plotly_white"
        plot_bgcolor = "white"  # Light background for subplots
        paper_bgcolor = "white"
        gridcolor = plot_bgcolor  # Light grid color for white mode
        font_color = "#000000"

    if zoom:
        shared_yaxes = False
    else:
        shared_yaxes = True

    if not wavelengths:
        wavelengths = [plate.wells[0].measurements[0].wavelength]

    if not isinstance(wavelengths, list):
        wavelengths = [wavelengths]

    fig = make_subplots(
        rows=8,
        cols=12,
        shared_xaxes=True,
        subplot_titles=_generate_possible_well_ids(),
        shared_yaxes=shared_yaxes,
    )
    colors = px.colors.qualitative.Plotly

    for well in plate.wells:
        for measurement, color in zip(well.measurements, colors):
            if measurement.wavelength not in wavelengths:
                continue

            fig.add_trace(
                go.Scatter(
                    x=measurement.time,
                    y=measurement.absorption,
                    name=f"{measurement.wavelength} nm",
                    mode="lines",
                    showlegend=False,
                    line=dict(color=color),
                    hovertemplate="%{y:.2f}<br>",
                ),
                col=well.x_pos + 1,
                row=well.y_pos + 1,
            )

    # Update x and y axes for dark mode or light mode
    fig.update_xaxes(
        showticklabels=False, gridcolor=gridcolor, zeroline=False, showline=False
    )
    fig.update_yaxes(
        showticklabels=False, gridcolor=gridcolor, zeroline=False, showline=False
    )

    # Update subplot backgrounds and layout
    fig.update_layout(
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=dict(color=font_color),
        hovermode="x",
        title=dict(
            text=name,
            font=dict(color=font_color),
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        template=theme,
    )

    if static:
        fig.show("png")

    # fig.write_json("plate_visualization.json")

    fig.show()


def _generate_possible_well_ids() -> list[str]:
    characters = "ABCDEFGH"
    integers = range(1, 13)  # 1 to 12

    sub_char = characters[:8]
    sub_int = integers[:12]

    # Generate combinations of characters and integers
    combinations = ["".join(item) for item in it.product(sub_char, map(str, sub_int))]

    return combinations
