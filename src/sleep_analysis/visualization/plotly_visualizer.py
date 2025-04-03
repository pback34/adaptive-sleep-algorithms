"""
Plotly implementation of the visualization interface.

This module provides a concrete implementation of VisualizerBase using the Plotly library.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Import the necessary Plotly components
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "Plotly is required for PlotlyVisualizer. "
        "Install it with 'pip install plotly'."
    )

from .base import VisualizerBase
from ..signals.eeg_sleep_stage_signal import EEGSleepStageSignal # Added import

class PlotlyVisualizer(VisualizerBase):
    """
    Plotly implementation of the visualization interface.
    
    This class provides concrete implementations of all abstract methods
    defined in VisualizerBase using the Plotly library.
    """
    
    def __init__(self, default_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Plotly visualizer with default parameters.
        
        Args:
            default_params: Optional dictionary of default visualization parameters
        """
        default_plotly_params = {
            'width': 800,
            'height': 500,
            'template': 'plotly_white',
            'colormap': 'viridis',
            'palette': 'Plotly',  # Add default palette for consistency with Bokeh
            'colorscale': 'Plotly'  # Default colorscale for qualitative data
        }
        
        # Merge provided defaults with Plotly-specific defaults
        super().__init__(default_params={**default_plotly_params, **(default_params or {})})
    
    def create_figure(self, **kwargs) -> Any:
        """
        Create a new Plotly figure.
        
        Args:
            **kwargs: Optional parameters for figure creation
                - width: Figure width in pixels (optional)
                - height: Figure height in pixels (optional)
                - template: Plotly template to use
                - title: Figure title
                
        Returns:
            A Plotly figure object
        """
        # Create the figure
        fig = go.Figure()
        
        # Update layout with settings
        layout_updates = {
            'template': kwargs.get('template', self.default_params.get('template', 'plotly_white')),
            'title': kwargs.get('title', ''),
            'hovermode': 'closest'
        }
        
        # Only set width and height if explicitly provided in kwargs
        if 'width' in kwargs:
            layout_updates['width'] = kwargs['width']
        if 'height' in kwargs:
            layout_updates['height'] = kwargs['height']
        
        # Set autosize based on whether width and height are provided
        if 'width' in layout_updates and 'height' in layout_updates:
            layout_updates['autosize'] = False
        else:
            layout_updates['autosize'] = True
        
        fig.update_layout(**layout_updates)
        
        # Set axis types if specified
        x_axis_type = kwargs.get('x_axis_type')
        y_axis_type = kwargs.get('y_axis_type')

        if x_axis_type:
            # Map 'datetime' to 'date' for Plotly
            plotly_x_type = 'date' if x_axis_type == 'datetime' else x_axis_type
            fig.update_xaxes(type=plotly_x_type)
        if y_axis_type:
            fig.update_yaxes(type=y_axis_type)
        
        return fig
    
    def add_line_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """
        Add a line plot to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the plot to
            x: X-axis data
            y: Y-axis data
            **kwargs: Optional styling parameters
                - line_color: Color of the line
                - line_width: Width of the line
                - line_dash: Dash pattern ('solid', 'dash', etc.)
                - opacity: Transparency (0 to 1)
                - name: Name for the legend
                
        Returns:
            The Plotly trace that was added
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract styling parameters
        if 'color_index' in params:
            import plotly.express as px
            # Get a color from a Plotly qualitative colorscale
            colorscale = params.get('colorscale', self.default_params.get('colorscale', 'Plotly'))
            if hasattr(px.colors.qualitative, colorscale):
                colors = getattr(px.colors.qualitative, colorscale)
                color_idx = params['color_index'] % len(colors)
                line_color = params.get('line_color', params.get('color', colors[color_idx]))
            else:
                line_color = params.get('line_color', params.get('color', 'blue'))
        else:
            line_color = params.get('line_color', params.get('color', 'blue'))
        line_width = params.get('line_width', 2)
        line_dash = params.get('line_dash', 'solid')
        opacity = params.get('opacity', params.get('alpha', 1.0))
        name = params.get('name', None)
        
        # Clean data (handle NaN values)
        x_data = x
        y_data = y
        
        if isinstance(x, pd.Series):
            x_data = x.values
        if isinstance(y, pd.Series):
            y_data = y.values
            
        # Create the trace
        trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            line=dict(
                color=line_color,
                width=line_width,
                dash=line_dash
            ),
            opacity=opacity,
            name=name,
            showlegend=name is not None,
            line_shape=params.get('line_shape', 'linear') # Add line_shape parameter
        )

        # Add the trace to the figure
        figure.add_trace(trace)

        # Handle categorical y-axis mapping if provided
        y_axis_mapping = params.get('y_axis_mapping')
        if y_axis_mapping:
            num_to_category = y_axis_mapping
            tickvals = sorted(list(num_to_category.keys())) # Ensure ticks are sorted
            ticktext = [num_to_category[i] for i in tickvals]
            figure.update_yaxes(tickvals=tickvals, ticktext=ticktext)

            # Adjust hover template for categorical data
            # Create hover text mapping numbers back to categories
            category_names = [num_to_category.get(val, 'Unknown') for val in y_data]
            hover_template = '<b>Time</b>: %{x}<br><b>Stage</b>: %{customdata}<extra></extra>'
            figure.update_traces(customdata=category_names, hovertemplate=hover_template)
        elif params.get('hover_text'):
             # Default hover text for numerical data
             figure.update_traces(
                 hovertext=params['hover_text'],
                 hoverinfo='text'
             )
        else:
             # Default hover template for numerical data if no specific text provided
             hover_template = '<b>Time</b>: %{x}<br><b>Value</b>: %{y}<extra></extra>'
             figure.update_traces(hovertemplate=hover_template)
        
        return trace
    
    def add_scatter_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """
        Add a scatter plot to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the plot to
            x: X-axis data
            y: Y-axis data
            **kwargs: Optional styling parameters
                - marker_color: Color of the markers
                - marker_size: Size of the markers
                - marker_symbol: Symbol for markers
                - opacity: Transparency (0 to 1)
                - name: Name for the legend
                
        Returns:
            The Plotly trace that was added
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract styling parameters
        marker_color = params.get('marker_color', params.get('color', 'blue'))
        marker_size = params.get('marker_size', params.get('size', 10))
        marker_symbol = params.get('marker_symbol', params.get('marker', 'circle'))
        opacity = params.get('opacity', params.get('alpha', 0.7))
        name = params.get('name', None)
        
        # Clean data (handle NaN values)
        x_data = x
        y_data = y
        
        if isinstance(x, pd.Series):
            x_data = x.values
        if isinstance(y, pd.Series):
            y_data = y.values
            
        # Create the trace
        trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                color=marker_color,
                size=marker_size,
                symbol=marker_symbol,
                opacity=opacity
            ),
            name=name,
            showlegend=name is not None
        )
        
        # Add the trace to the figure
        figure.add_trace(trace)
        
        # Add hover text if specified
        if params.get('hover_text'):
            figure.update_traces(
                hovertext=params['hover_text'],
                hoverinfo='text'
            )
        
        return trace
    
    def add_heatmap(self, figure: Any, data: Any, **kwargs) -> Any:
        """
        Add a heatmap to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the heatmap to
            data: 2D data array (numpy array or pandas DataFrame)
            **kwargs: Optional styling parameters
                - colorscale: Color scale to use
                - x_values: Values for the x-axis ticks
                - y_values: Values for the y-axis ticks
                - opacity: Transparency (0 to 1)
                
        Returns:
            The Plotly trace that was added
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract styling parameters
        colorscale = params.get('colorscale', params.get('colormap', 'viridis'))
        opacity = params.get('opacity', params.get('alpha', 1.0))
        
        # Prepare the data
        if isinstance(data, pd.DataFrame):
            z_data = data.values
            x_values = params.get('x_values', data.columns)
            y_values = params.get('y_values', data.index)
        else:
            z_data = data
            x_values = params.get('x_values', None)
            y_values = params.get('y_values', None)
            
        # Create the heatmap trace
        trace = go.Heatmap(
            z=z_data,
            x=x_values,
            y=y_values,
            colorscale=colorscale,
            opacity=opacity,
            showscale=params.get('show_colorbar', True),
            colorbar=dict(
                title=params.get('colorbar_title', '')
            )
        )
        
        # Add the trace to the figure
        figure.add_trace(trace)
        
        # Update layout for better display
        figure.update_layout(
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed' if params.get('reverse_y', False) else True)
        )
        
        return trace
    
    def create_grid_layout(self, figures: List[Any], rows: int, cols: int, **kwargs) -> Any:
        """
        Create a grid layout of multiple Plotly figures.
        
        Args:
            figures: List of Plotly figures to arrange in a grid
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            **kwargs: Optional layout parameters
                - link_x_axes: Whether to link x-axes across subplots (default: True)
                - shared_yaxes: Whether to share y-axes across subplots
                - subplot_titles: List of titles for each subplot
                - width: Grid width in pixels (optional)
                - height: Grid height in pixels (optional)
                
        Returns:
            A Plotly figure with the arranged subplots
        """
        # Extract layout parameters
        link_x_axes = kwargs.get('link_x_axes', True)
        shared_yaxes = kwargs.get('shared_yaxes', False)
        
        # Use provided subplot titles or extract from figures
        subplot_titles = kwargs.get('subplot_titles', None)
        if not subplot_titles:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug("No subplot_titles provided, extracting from figure titles")
            subplot_titles = [fig.layout.title.text for fig in figures if hasattr(fig.layout.title, 'text')]
            # Ensure we have enough titles
            subplot_titles.extend([''] * (len(figures) - len(subplot_titles)))
        else:
            # Ensure we have enough titles if subplot_titles was provided
            subplot_titles = subplot_titles[:len(figures)]
            if len(subplot_titles) < len(figures):
                subplot_titles.extend([''] * (len(figures) - len(subplot_titles)))
        
        # Create the subplot grid
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=link_x_axes,  # Link x-axes dynamically if True
            shared_yaxes=shared_yaxes,
            subplot_titles=subplot_titles[:len(figures)]
        )
        
        # Add each figure's traces to the grid
        for i, source_fig in enumerate(figures):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Add each trace from the source figure
            for trace in source_fig.data:
                fig.add_trace(trace, row=row, col=col)
            
            # Copy over axis settings
            for axis_prop in ['title', 'type', 'range']:
                # X-axis properties
                if hasattr(source_fig.layout.xaxis, axis_prop):
                    xaxis_key = f'xaxis{i+1}' if i > 0 else 'xaxis'
                    if not hasattr(fig.layout, xaxis_key):
                        continue
                    setattr(getattr(fig.layout, xaxis_key), axis_prop, 
                            getattr(source_fig.layout.xaxis, axis_prop))
                
                # Y-axis properties
                if hasattr(source_fig.layout.yaxis, axis_prop):
                    yaxis_key = f'yaxis{i+1}' if i > 0 else 'yaxis'
                    if not hasattr(fig.layout, yaxis_key):
                        continue
                    setattr(getattr(fig.layout, yaxis_key), axis_prop, 
                            getattr(source_fig.layout.yaxis, axis_prop))
                            
            # Copy shapes (for categorical regions, annotations, etc.)
            if hasattr(source_fig.layout, 'shapes') and source_fig.layout.shapes:
                 # Need to update shape references to the new subplot axes
                 current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                 for shape in source_fig.layout.shapes:
                     # Explicitly copy attributes instead of using dict(shape)
                     shape_dict = {
                         'type': shape.type,
                         'xref': f'x{i+1}', # Update xref immediately
                         'yref': f'y{i+1}' if shape.yref != 'paper' else 'paper', # Update yref immediately
                         'x0': shape.x0,
                         'y0': shape.y0,
                         'x1': shape.x1,
                         'y1': shape.y1,
                         'fillcolor': shape.fillcolor,
                         'opacity': shape.opacity,
                         'layer': shape.layer,
                         'line_width': shape.line.width if shape.line else 0,
                         'line_color': shape.line.color if shape.line else None,
                         'line_dash': shape.line.dash if shape.line else None,
                         # Copy other relevant attributes if needed
                     }
                     current_shapes.append(go.layout.Shape(shape_dict))
                 fig.layout.shapes = tuple(current_shapes)

            # Copy annotations (if any)
            if hasattr(source_fig.layout, 'annotations') and source_fig.layout.annotations:
                 current_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
                 for ann in source_fig.layout.annotations:
                     # Explicitly copy attributes instead of using dict(ann)
                     ann_dict = {
                         'xref': f'x{i+1}', # Update xref
                         'yref': f'y{i+1}', # Update yref
                         'x': ann.x,
                         'y': ann.y,
                         'ax': ann.ax,
                         'ay': ann.ay,
                         'xanchor': ann.xanchor,
                         'yanchor': ann.yanchor,
                         'text': ann.text,
                         'showarrow': ann.showarrow,
                         'arrowhead': ann.arrowhead,
                         'arrowsize': ann.arrowsize,
                         'arrowwidth': ann.arrowwidth,
                         'arrowcolor': ann.arrowcolor,
                         'font': ann.font.to_plotly_json() if ann.font else None, # Copy font dict safely
                         'align': ann.align,
                         'valign': ann.valign,
                         'bgcolor': ann.bgcolor,
                         'bordercolor': ann.bordercolor,
                         'borderwidth': ann.borderwidth,
                         'borderpad': ann.borderpad,
                         'opacity': ann.opacity,
                         # Copy other relevant attributes if needed
                     }
                     current_annotations.append(go.layout.Annotation(ann_dict))
                 fig.layout.annotations = tuple(current_annotations)
        
        # Update layout
        layout_updates = {
            'template': kwargs.get('template', self.default_params.get('template', 'plotly_white')),
            'title': kwargs.get('title', '')
        }
        
        # Only set width and height if explicitly provided in kwargs
        if 'width' in kwargs:
            layout_updates['width'] = kwargs['width']
        if 'height' in kwargs:
            layout_updates['height'] = kwargs['height']
        
        # Set autosize based on whether width and height are provided
        if 'width' in layout_updates and 'height' in layout_updates:
            layout_updates['autosize'] = False
        else:
            layout_updates['autosize'] = True
        
        fig.update_layout(**layout_updates)
        
        return fig
    
    def add_legend(self, figure: Any, **kwargs) -> None:
        """
        Add or customize the legend of a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the legend to
            **kwargs: Optional legend parameters
                - orientation: Legend orientation ('h' for horizontal, 'v' for vertical)
                - x: X-position of the legend
                - y: Y-position of the legend
                - bgcolor: Background color
                - bordercolor: Border color
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract legend parameters
        orientation = params.get('legend_orientation', 'v')
        x = params.get('legend_x', 1.02)
        y = params.get('legend_y', 1)
        bgcolor = params.get('legend_bgcolor', 'rgba(255, 255, 255, 0.5)')
        bordercolor = params.get('legend_bordercolor', 'rgba(0, 0, 0, 0.2)')
        
        # Update legend
        figure.update_layout(
            showlegend=True,
            legend=dict(
                orientation=orientation,
                x=x,
                y=y,
                bgcolor=bgcolor,
                bordercolor=bordercolor
            )
        )
    
    def set_title(self, figure: Any, title: str, **kwargs) -> None:
        """
        Set the title of a Plotly figure.
        
        Args:
            figure: The Plotly figure to set the title for
            title: The title text
            **kwargs: Optional title styling parameters
                - font_size: Font size
                - font_family: Font family
                - x: X-position of the title
                - y: Y-position of the title
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract title parameters
        font_size = params.get('title_font_size', 18)
        font_family = params.get('title_font_family', 'Arial, sans-serif')
        x = params.get('title_x', 0.5)
        y = params.get('title_y', 0.9)
        
        # Set the title
        figure.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=font_size,
                    family=font_family
                ),
                x=x,
                y=y
            )
        )
    
    def set_axis_labels(self, figure: Any, x_label: Optional[str] = None, 
                        y_label: Optional[str] = None, **kwargs) -> None:
        """
        Set axis labels for a Plotly figure.
        
        Args:
            figure: The Plotly figure to set axis labels for
            x_label: Optional label for the x-axis
            y_label: Optional label for the y-axis
            **kwargs: Optional label styling parameters
                - font_size: Font size for axis labels
                - font_family: Font family for axis labels
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract label parameters
        font_size = params.get('axis_label_font_size', 14)
        font_family = params.get('axis_label_font_family', 'Arial, sans-serif')
        
        # Set x-axis label if provided
        if x_label is not None:
            figure.update_xaxes(
                title_text=x_label,
                title_font=dict(
                    size=font_size,
                    family=font_family
                )
            )
            
        # Set y-axis label if provided
        if y_label is not None:
            figure.update_yaxes(
                title_text=y_label,
                title_font=dict(
                    size=font_size,
                    family=font_family
                )
            )
    
    def set_axis_limits(self, figure: Any, x_min: Optional[Any] = None, 
                        x_max: Optional[Any] = None, y_min: Optional[Any] = None, 
                        y_max: Optional[Any] = None) -> None:
        """
        Set axis limits for a Plotly figure.
        
        Args:
            figure: The Plotly figure to set axis limits for
            x_min: Optional minimum value for the x-axis
            x_max: Optional maximum value for the x-axis
            y_min: Optional minimum value for the y-axis
            y_max: Optional maximum value for the y-axis
        """
        # Set x-axis range if provided
        if x_min is not None or x_max is not None:
            # Get current range
            current_range = figure.layout.xaxis.range if hasattr(figure.layout.xaxis, 'range') else [None, None]
            
            # Update with new values
            new_range = [
                x_min if x_min is not None else current_range[0],
                x_max if x_max is not None else current_range[1]
            ]
            
            # Only set if both values are not None
            if all(val is not None for val in new_range):
                figure.update_xaxes(range=new_range)
            
        # Set y-axis range if provided
        if y_min is not None or y_max is not None:
            # Get current range
            current_range = figure.layout.yaxis.range if hasattr(figure.layout.yaxis, 'range') else [None, None]
            
            # Update with new values
            new_range = [
                y_min if y_min is not None else current_range[0],
                y_max if y_max is not None else current_range[1]
            ]
            
            # Only set if both values are not None
            if all(val is not None for val in new_range):
                figure.update_yaxes(range=new_range)
    
    def add_hover_tooltip(self, figure: Any, tooltips: Any, **kwargs) -> None:
        """
        Add hover tooltips to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add tooltips to
            tooltips: List of field names to include in hover tooltip
            **kwargs: Optional tooltip parameters
                - mode: Hover mode ('closest', 'x', 'y', 'x unified', 'y unified')
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract tooltip parameters
        mode = params.get('hover_mode', 'closest')
        
        # Update hovermode
        figure.update_layout(hovermode=mode)
        
        # If tooltips is a dictionary, use it for hovertemplate
        if isinstance(tooltips, dict):
            hover_template = '<br>'.join([f"{k}: {v}" for k, v in tooltips.items()])
            figure.update_traces(hovertemplate=hover_template)
        # If tooltips is a list, use it for hoverlabel
        elif isinstance(tooltips, list):
            figure.update_traces(hoverinfo='text', hovertext=tooltips)
    
    def add_vertical_line(self, figure: Any, x: Any, **kwargs) -> Any:
        """
        Add a vertical line to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the line to
            x: The x-coordinate of the line
            **kwargs: Optional line styling parameters
                - color: Line color
                - width: Line width
                - dash: Line dash pattern
                - opacity: Line transparency
                
        Returns:
            The Plotly shape that was added
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract line parameters
        color = params.get('line_color', params.get('color', 'red'))
        width = params.get('line_width', params.get('width', 1))
        dash = params.get('line_dash', params.get('dash', 'solid'))
        opacity = params.get('opacity', params.get('alpha', 0.7))
        
        # Create vertical line shape
        vline = dict(
            type='line',
            xref='x',
            yref='paper',
            x0=x,
            y0=0,
            x1=x,
            y1=1,
            line=dict(
                color=color,
                width=width,
                dash=dash
            ),
            opacity=opacity
        )
        
        # Add shape to the figure
        if 'shapes' in figure.layout:
            figure.layout.shapes = list(figure.layout.shapes) + [vline]
        else:
            figure.update_layout(shapes=[vline])
        
        return vline
    
    def add_horizontal_line(self, figure: Any, y: Any, **kwargs) -> Any:
        """
        Add a horizontal line to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the line to
            y: The y-coordinate of the line
            **kwargs: Optional line styling parameters
                - color: Line color
                - width: Line width
                - dash: Line dash pattern
                - opacity: Line transparency
                
        Returns:
            The Plotly shape that was added
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract line parameters
        color = params.get('line_color', params.get('color', 'green'))
        width = params.get('line_width', params.get('width', 1))
        dash = params.get('line_dash', params.get('dash', 'solid'))
        opacity = params.get('opacity', params.get('alpha', 0.7))
        
        # Create horizontal line shape
        hline = dict(
            type='line',
            xref='paper',
            yref='y',
            x0=0,
            y0=y,
            x1=1,
            y1=y,
            line=dict(
                color=color,
                width=width,
                dash=dash
            ),
            opacity=opacity
        )
        
        # Add shape to the figure
        if 'shapes' in figure.layout:
            figure.layout.shapes = list(figure.layout.shapes) + [hline]
        else:
            figure.update_layout(shapes=[hline])
        
        return hline
    
    def add_region(self, figure: Any, x_start: Any, x_end: Any, **kwargs) -> Any:
        """
        Add a highlighted region to a Plotly figure.
        
        Args:
            figure: The Plotly figure to add the region to
            x_start: The starting x-coordinate of the region
            x_end: The ending x-coordinate of the region
            **kwargs: Optional region styling parameters
                - color: Fill color
                - opacity: Fill transparency
                - name: Name for the legend
                
        Returns:
            The Plotly shape that was added
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract region parameters
        color = params.get('fill_color', params.get('color', 'yellow'))
        opacity = params.get('opacity', params.get('alpha', 0.2))
        
        # Create region shape
        region = dict(
            type='rect',
            xref='x',
            yref='paper',
            x0=x_start,
            y0=0,
            x1=x_end,
            y1=1,
            fillcolor=color,
            opacity=opacity,
            layer='below',
            line_width=0
        )
        
        # Add shape to the figure
        if 'shapes' in figure.layout:
            figure.layout.shapes = list(figure.layout.shapes) + [region]
        else:
            figure.update_layout(shapes=[region])
        
        # Add annotation if name is provided
        if 'name' in params:
            figure.add_annotation(
                x=(x_start + x_end) / 2,
                y=1.05,
                text=params['name'],
                showarrow=False,
                font=dict(color=color)
            )
        
        return region
        
    def add_categorical_regions(self, figure: Any, start_times: Any, end_times: Any, 
                                categories: Any, category_map: Dict[str, str], 
                                **kwargs) -> List[Any]:
        """
        Add colored regions for categorical data using shapes.
        """
        regions = []
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        alpha = params.get('alpha', 0.3)
        y_range = params.get('y_range', (0, 1)) # Default y-range for regions
        
        shapes = []
        annotations = []
        legend_items = {} # To track items for potential manual legend

        # Add a shape for each interval
        for start, end, cat in zip(start_times, end_times, categories):
             if pd.isna(start) or pd.isna(end):
                 continue
                 
             color = category_map.get(cat, 'gray') # Default to gray
             
             shape = dict(
                 type='rect',
                 xref='x',
                 yref='paper', # Span full y-axis
                 x0=start,
                 y0=y_range[0],
                 x1=end,
                 y1=y_range[1],
                 fillcolor=color,
                 opacity=alpha,
                 layer='below',
                 line_width=0
             )
             shapes.append(shape)
             
             # Store info for legend if needed
             if cat not in legend_items:
                 legend_items[cat] = color

        # Add all shapes at once
        figure.update_layout(shapes=shapes)
        
        # Optionally add legend (Plotly doesn't directly support legends for shapes)
        # We can add dummy traces or annotations as a workaround
        if params.get('add_legend', False):
            for category, color in legend_items.items():
                 # Add dummy scatter trace for legend
                 figure.add_trace(go.Scatter(
                     x=[None], y=[None], # No actual data
                     mode='markers',
                     marker=dict(color=color, size=10),
                     name=category,
                     showlegend=True
                 ))

        return shapes # Return the list of shape dicts

    def save(self, figure: Any, filename: str, format: str = "html", **kwargs) -> None:
        """
        Save a Plotly figure to a file.
        
        Args:
            figure: The Plotly figure to save
            filename: The output filename
            format: The output format (html, png, jpg, svg, pdf)
            **kwargs: Optional export parameters
                - include_plotlyjs: How to include plotly.js (True, False, 'cdn')
                - full_html: Whether to include full HTML wrapper
                - width: Override width in pixels (optional)
                - height: Override height in pixels (optional)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Check if figure has data/traces
        if hasattr(figure, 'data') and not figure.data:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Figure has no data traces, adding a placeholder message")
            figure.add_annotation(
                text="No data to display",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20, color="red")
            )
        
        # Handle different output formats
        format = format.lower()
        
        if format in ('html', 'htm'):
            # Extract HTML parameters
            include_plotlyjs = kwargs.get('include_plotlyjs', self.default_params.get('include_plotlyjs', 'cdn'))
            full_html = kwargs.get('full_html', self.default_params.get('full_html', True))
            
            # Only update layout dimensions if explicitly provided in kwargs
            layout_updates = {}
            if 'width' in kwargs:
                layout_updates['width'] = kwargs['width']
            if 'height' in kwargs:
                layout_updates['height'] = kwargs['height']
            
            if layout_updates:
                figure.update_layout(**layout_updates, autosize=False)
                
                # Log the dimensions being used
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Setting figure dimensions to width={layout_updates.get('width')}, height={layout_updates.get('height')}")
            
            # Set config based on whether the figure has fixed dimensions
            # This ensures responsive=True when no dimensions are specified
            has_width = figure.layout.width is not None
            has_height = figure.layout.height is not None
            is_fixed_size = has_width and has_height
            
            config = {
                'displayModeBar': True,
                'scrollZoom': True,
                'responsive': not is_fixed_size
            }
            
            # Write to HTML
            figure.write_html(
                filename,
                include_plotlyjs=include_plotlyjs,
                full_html=full_html,
                config=config
            )
            
        elif format in ('png', 'jpg', 'jpeg', 'webp'):
            # Combine default parameters with provided kwargs
            params = {**self.default_params, **kwargs}
            
            # Extract image parameters
            scale = params.get('scale', 1.0)
            width = params.get('width')
            height = params.get('height')
            
            # Write to image
            figure.write_image(
                filename,
                scale=scale,
                width=width,
                height=height
            )
            
        elif format in ('svg'):
            figure.write_image(filename)
            
        elif format in ('pdf'):
            figure.write_image(filename)
            
        elif format in ('json'):
            figure.write_json(filename)
            
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def show(self, figure: Any) -> None:
        """
        Display a Plotly figure.
        
        Args:
            figure: The Plotly figure to display
        """
        figure.show()

    def visualize_hypnogram(self, figure: Any, signal: EEGSleepStageSignal, **kwargs) -> Any:
        """
        Create a sleep stage hypnogram visualization using Plotly filled scatters.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Merge default parameters with provided kwargs
        params = {**self.default_params, **kwargs}

        # Extract data from signal
        df = signal.get_data()
        stage_column = 'sleep_stage'
        if stage_column not in df.columns:
            logger.warning("Sleep stage column not found in data")
            return None

        # Define stage order and mapping to numeric values (include 'Unknown')
        default_order = ['Awake', 'REM', 'N1', 'N2', 'N3', 'Unknown']
        stage_order = params.get('stage_order', default_order)
        stage_to_num = {stage: i for i, stage in enumerate(stage_order)}
        num_stages = len(stage_order)

        # Define colors for each stage (fallback to default palette if not provided)
        default_colors = {
            'Awake': '#FF5733',  # Orange-red
            'N1': '#33A8FF',     # Light blue
            'N2': '#3358FF',     # Medium blue
            'N3': '#0D0C8A',     # Dark blue
            'REM': '#AA33FF',    # Purple
            'Unknown': '#CCCCCC' # Grey for Unknown/Other
        }
        # Ensure default colors cover the default order
        colorscale = params.get('colorscale', self.default_params.get('colorscale', 'Plotly'))
        if hasattr(px.colors.qualitative, colorscale):
             palette = getattr(px.colors.qualitative, colorscale)
        else:
             palette = px.colors.qualitative.Plotly # Fallback palette

        for i, stage in enumerate(stage_order):
             if stage not in default_colors:
                 default_colors[stage] = palette[i % len(palette)]

        stage_colors = params.get('stage_colors', default_colors)

        # --- Data Processing for Plotly Filled Area ---
        # Create a timeline with start/end points for each segment
        timeline = []
        for i in range(len(df)):
            stage = df[stage_column].iloc[i]
            # Map unexpected stages to 'Unknown'
            if stage not in stage_to_num:
                logger.debug(f"Mapping stage '{stage}' to 'Unknown'")
                stage = 'Unknown'

            stage_num = stage_to_num[stage]
            time = df.index[i]

            # Add point for this time
            timeline.append({'time': time, 'stage': stage, 'stage_num': stage_num})

            # Add point for the *next* time step to define the segment end
            # This creates the step-like appearance
            if i < len(df) - 1:
                next_time = df.index[i+1]
                timeline.append({'time': next_time, 'stage': stage, 'stage_num': stage_num})
            else:
                # For the last segment, estimate end time (e.g., add typical epoch duration)
                # Or use the last timestamp if duration is not critical for visualization end
                # Here, we just duplicate the last point's time for simplicity
                 timeline.append({'time': time, 'stage': stage, 'stage_num': stage_num})


        # Sort timeline by time (important for plotting)
        timeline = sorted(timeline, key=lambda x: x['time'])

        # Create processed data structure for plotting
        processed_data = {'time': [item['time'] for item in timeline]}
        for stage in stage_order:
            # Create a column for each stage: height = num_stages - stage_num when active, 0 otherwise
            # This plots 'Awake' at the top (y=num_stages), 'N3' near the bottom (y=1)
            processed_data[stage] = [
                (num_stages - item['stage_num']) if item['stage'] == stage else 0
                for item in timeline
            ]
        # --- End Data Processing ---

        # Add filled area traces for each stage
        traces = []
        for stage in stage_order:
            trace = go.Scatter(
                x=processed_data['time'],
                y=processed_data[stage],
                mode='lines',
                line=dict(width=0), # No line, just fill
                fill='tozeroy',     # Fill down to y=0
                fillcolor=stage_colors.get(stage, '#CCCCCC'),
                name=stage,
                hoverinfo='name+x', # Show stage name and time on hover
                hoverlabel=dict(namelength=-1),
                showlegend=True # Ensure legend items are created
            )
            figure.add_trace(trace)
            traces.append(trace)

        # Update y-axis to show stage names correctly ordered
        # Tick values correspond to the 'height' calculated (1 to num_stages)
        # Tick text corresponds to the stage names in the desired order
        figure.update_yaxes(
            tickvals=list(range(1, num_stages + 1)),
            ticktext=list(reversed(stage_order)), # Reverse order to match plotting height
            range=[0, num_stages + 0.5] # Set range from 0 to slightly above top stage
        )

        return traces # Return the list of traces added

    def _add_statistics_annotation(self, figure: Any, stats_text: List[str], **kwargs) -> None:
        """Add sleep statistics annotation using Plotly annotations."""
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}

        # Extract annotation parameters
        x = params.get('stats_x', 0.01)
        y = params.get('stats_y', 0.01)
        xref = params.get('stats_xref', 'paper')
        yref = params.get('stats_yref', 'paper')
        align = params.get('stats_align', 'left')
        font_size_param = params.get('stats_font_size', 10) # Get the parameter
        bgcolor = params.get('stats_bg_color', 'rgba(255, 255, 255, 0.8)')
        border_color = params.get('stats_border_color', 'black')
        border_width = params.get('stats_border_width', 1)
        border_pad = params.get('stats_border_pad', 4)

        # Join the list of strings with HTML line breaks for Plotly
        annotation_text = "<br>".join(stats_text)

        # --- Convert font size to number ---
        font_size = 10 # Default numeric font size
        if isinstance(font_size_param, (int, float)):
            font_size = font_size_param
        elif isinstance(font_size_param, str):
            try:
                # Remove 'pt', 'px', etc. and convert to float then int
                font_size = int(float(font_size_param.lower().replace('pt', '').replace('px', '').strip()))
            except ValueError:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not parse stats_font_size '{font_size_param}' as number. Using default {font_size}.")
        # --- End font size conversion ---

        # Add the annotation
        figure.add_annotation(
            xref=xref, yref=yref,
            x=x, y=y,
            text=annotation_text,
            showarrow=False,
            font=dict(size=font_size),
            align=align,
            bgcolor=bgcolor,
            bordercolor=border_color,
            borderwidth=border_width,
            borderpad=border_pad
        )
