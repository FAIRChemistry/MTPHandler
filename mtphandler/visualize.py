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

    # Dynamically determine grid size based on actual well positions
    max_row = max(well.y_pos for well in plate.wells) + 1
    max_col = max(well.x_pos for well in plate.wells) + 1

    # Use standard plate dimensions as minimum, but expand if needed
    rows = max(8, max_row)
    cols = max(12, max_col)

    # Generate well IDs for the actual grid size
    well_ids = _generate_well_ids_for_grid(rows, cols)

    try:
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            subplot_titles=well_ids,
            shared_yaxes=shared_yaxes,
        )
    except Exception:
        # Fallback: create a grid that exactly fits the data
        print(
            f"Warning: Could not create {rows}x{cols} grid, falling back to minimal grid"
        )
        print(f"Well positions range: rows 0-{max_row - 1}, cols 0-{max_col - 1}")
        rows, cols = max_row, max_col
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            subplot_titles=_generate_well_ids_for_grid(rows, cols),
            shared_yaxes=shared_yaxes,
        )
    colors = px.colors.qualitative.Plotly

    # Detect if this is endpoint data (single timepoint)
    is_endpoint_data = len(plate.times) == 1 and all(
        len(measurement.time) == 1
        for well in plate.wells
        for measurement in well.measurements
    )

    # For endpoint data, use heatmap visualization instead
    if is_endpoint_data:
        visualize_plate_heatmap(
            plate=plate,
            name=name,
            wavelength=wavelengths[0] if wavelengths else None,
            darkmode=darkmode,
            log_scale=True,
        )
        return

    for well in plate.wells:
        for measurement, color in zip(well.measurements, colors):
            if measurement.wavelength not in wavelengths:
                continue

            try:
                if is_endpoint_data:
                    # For endpoint data, use bar chart
                    fig.add_trace(
                        go.Bar(
                            x=[well.id],  # Well ID as x-axis
                            y=measurement.absorption,
                            name=f"{measurement.wavelength} nm"
                            if measurement.wavelength != 0
                            else "Luminescence",
                            showlegend=False,
                            marker=dict(color=color),
                            hovertemplate=f"<b>{well.id}</b><br>Value: %{{y:.0f}}<extra></extra>",
                        ),
                        col=well.x_pos + 1,
                        row=well.y_pos + 1,
                    )
                else:
                    # For kinetic data, use line plot
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
            except Exception as e:
                print(
                    f"Warning: Could not plot well {well.id} at position ({well.y_pos + 1}, {well.x_pos + 1}): {e}"
                )
                continue

    # Update x and y axes for dark mode or light mode
    if is_endpoint_data:
        # For endpoint data, show y-axis labels to see values
        fig.update_xaxes(
            showticklabels=False, gridcolor=gridcolor, zeroline=False, showline=False
        )
        fig.update_yaxes(
            showticklabels=True,
            gridcolor=gridcolor,
            zeroline=True,
            showline=True,
            tickformat=".0f",  # Format as integers for large values
        )
    else:
        # For kinetic data, hide labels as before
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


def visualize_plate_heatmap(
    plate: Plate,
    name: str,
    wavelength: float | None = None,
    darkmode: bool = False,
    log_scale: bool = False,
):
    """Visualize endpoint plate data as a heatmap."""

    if darkmode:
        theme = "plotly_dark"
        colorscale = "Viridis"
    else:
        theme = "plotly_white"
        colorscale = "Blues"

    # Get the wavelength to visualize
    if wavelength is None:
        wavelength = plate.wells[0].measurements[0].wavelength

    # Create matrix for heatmap
    max_row = max(well.y_pos for well in plate.wells) + 1
    max_col = max(well.x_pos for well in plate.wells) + 1

    # Initialize matrix with NaN
    matrix: list[list[float | None]] = [
        [None for _ in range(max_col)] for _ in range(max_row)
    ]
    well_ids = [["" for _ in range(max_col)] for _ in range(max_row)]

    # Fill matrix with data
    for well in plate.wells:
        for measurement in well.measurements:
            if measurement.wavelength == wavelength:
                matrix[well.y_pos][well.x_pos] = float(measurement.absorption[0])
                well_ids[well.y_pos][well.x_pos] = well.id

        # Apply log scaling if requested
    original_matrix: list[list[float | None]] | None = None

    if log_scale:
        import numpy as np

        # Create log-scaled matrix and keep original values
        log_matrix: list[list[float | None]] = []
        original_matrix = []

        for row_idx in range(max_row):
            log_row: list[float | None] = []
            orig_row: list[float | None] = []
            for col_idx in range(max_col):
                value = matrix[row_idx][col_idx]
                if value is not None:
                    original_value = float(value)
                    log_value = float(np.log10(original_value + 1))
                    log_row.append(log_value)
                    orig_row.append(original_value)
                else:
                    log_row.append(None)
                    orig_row.append(None)
            log_matrix.append(log_row)
            original_matrix.append(orig_row)

        matrix = log_matrix
        scale_title = "Log10"
        hover_template = "<b>%{text}</b><br>Log Value: %{z:.2f}<br>Original: %{customdata:.0f}<extra></extra>"
    else:
        scale_title = "Value"
        hover_template = "<b>%{text}</b><br>Value: %{z:.0f}<extra></extra>"

    # Create heatmap
    heatmap_data = go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        showscale=True,
        hovertemplate=hover_template,
        text=well_ids,
        colorbar=dict(title=scale_title),
    )

    if log_scale and original_matrix:
        heatmap_data.customdata = original_matrix

    fig = go.Figure(data=heatmap_data)

    # Create row and column labels
    row_labels = [chr(65 + i) for i in range(max_row)]  # A, B, C, ...
    col_labels = [str(i + 1) for i in range(max_col)]  # 1, 2, 3, ...

    fig.update_layout(
        title=f"{name}" if wavelength != 0 else f"{name}",
        xaxis=dict(title="Column", tickvals=list(range(max_col)), ticktext=col_labels),
        yaxis=dict(
            title="Row",
            tickvals=list(range(max_row)),
            ticktext=row_labels,
            autorange="reversed",
        ),
        template=theme,
    )

    fig.show()


def _generate_well_ids_for_grid(rows: int, cols: int) -> list[str]:
    """Generate well IDs for a grid of specified dimensions."""
    characters = "ABCDEFGHIJKLMNOP"  # Extended to support larger plates
    integers = range(1, cols + 1)

    sub_char = characters[:rows]
    sub_int = integers

    # Generate combinations of characters and integers
    combinations = ["".join(item) for item in it.product(sub_char, map(str, sub_int))]

    return combinations


def _generate_possible_well_ids() -> list[str]:
    """Generate well IDs for standard 8x12 plate (backward compatibility)."""
    return _generate_well_ids_for_grid(8, 12)
