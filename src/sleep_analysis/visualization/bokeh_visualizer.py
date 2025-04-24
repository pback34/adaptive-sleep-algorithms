"""
Bokeh implementation of the visualization interface.

This module provides a concrete implementation of VisualizerBase using the Bokeh library.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Get logger
logger = logging.getLogger(__name__)

# Import the necessary Bokeh components
try:
    from bokeh.plotting import figure, show, output_file
    from bokeh.models import (
        ColumnDataSource, HoverTool, LinearAxis, Range1d, 
        ColumnDataSource, HoverTool, LinearAxis, Range1d,
        Span, BoxAnnotation, Legend, LegendItem, GridBox, Label # Added Label
    )
    from bokeh.layouts import gridplot
    from bokeh.palettes import Category10_10, Category20 # Added Category20 for more colors
except ImportError:
    raise ImportError(
        "Bokeh is required for BokehVisualizer. "
        "Install it with 'pip install bokeh'."
    )

from .base import VisualizerBase
from ..signals.eeg_sleep_stage_signal import EEGSleepStageSignal # Added import

class BokehVisualizer(VisualizerBase):
    """
    Bokeh implementation of the visualization interface.
    
    This class provides concrete implementations of all abstract methods
    defined in VisualizerBase using the Bokeh library.
    """
    
    def __init__(self, default_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bokeh visualizer with default parameters.
        
        Args:
            default_params: Optional dictionary of default visualization parameters
        """
        default_bokeh_params = {
            'width': 800,
            'height': 400,
            'tools': 'pan,wheel_zoom,box_zoom,reset,save,hover',
            'palette': Category10_10
        }
        
        # Merge provided defaults with Bokeh-specific defaults
        super().__init__(default_params={**default_bokeh_params, **(default_params or {})})
    
    def create_figure(self, **kwargs) -> Any:
        """
        Create a new Bokeh figure.
        
        Args:
            **kwargs: Optional parameters for figure creation
                - width: Figure width in pixels
                - height: Figure height in pixels
                - tools: String of comma-separated tool names
                - x_axis_type: Type of x-axis ('datetime', 'linear', etc.)
                - y_axis_type: Type of y-axis ('datetime', 'linear', etc.)
                - title: Figure title
                - sizing_mode: How the figure resizes ('fixed', 'stretch_both', etc.)
                
        Returns:
            A Bokeh figure object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # If user did not specify sizing_mode, set it based on whether width and height are provided
        if 'sizing_mode' not in kwargs:
            if 'width' in kwargs and 'height' in kwargs:
                params['sizing_mode'] = 'fixed'
            else:
                params['sizing_mode'] = 'stretch_both'
        
        # Extract Bokeh-specific parameters
        width = params.get('width', 800)
        height = params.get('height', 400)
        tools = params.get('tools', 'pan,wheel_zoom,box_zoom,reset,save')
        x_axis_type = params.get('x_axis_type', 'datetime')  # Default to datetime for time series
        y_axis_type = params.get('y_axis_type', 'auto')
        title = params.get('title', '')
        sizing_mode = params.get('sizing_mode')
        
        # Create the figure
        fig = figure(
            width=width,
            height=height,
            tools=tools,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            title=title,
            sizing_mode=sizing_mode
        )
        
        return fig
    
    def add_line_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """
        Add a line plot to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the plot to
            x: X-axis data
            y: Y-axis data
            **kwargs: Optional styling parameters
                - line_color: Color of the line
                - line_width: Width of the line
                - line_dash: Dash pattern ('solid', 'dashed', etc.)
                - alpha: Transparency (0 to 1)
                - name: Name for the legend
                
        Returns:
            The Bokeh line renderer object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract styling parameters
        if 'color_index' in params and 'palette' in self.default_params:
            palette = self.default_params['palette']
            color_idx = params['color_index'] % len(palette)
            line_color = params.get('line_color', params.get('color', palette[color_idx]))
        else:
            line_color = params.get('line_color', params.get('color', 'blue'))
        line_width = params.get('line_width', 2)
        line_dash = params.get('line_dash', 'solid')
        alpha = params.get('alpha', 1.0)
        name = params.get('name', None)
        
        # Create a ColumnDataSource
        source_data = {'x': x, 'y': y}
        if isinstance(x, pd.Series):
            source_data['x'] = x.values
        if isinstance(y, pd.Series):
            source_data['y'] = y.values
            
        # Handle NaN values
        valid_mask = ~np.isnan(source_data['y'])
        if not valid_mask.all():
            source_data['x'] = source_data['x'][valid_mask]
            source_data['y'] = source_data['y'][valid_mask]
        
        # Check if we have any valid data points after filtering
        import logging
        logger = logging.getLogger(__name__)
        if len(source_data['x']) == 0:
            logger.warning(f"No valid data to plot for signal {kwargs.get('name', 'unknown')}, column {kwargs.get('col', 'unknown')}")
            return None
            
        source = ColumnDataSource(data=source_data)
        
        # Add the line
        line = figure.line(
            'x', 'y', 
            source=source,
            line_color=line_color, 
            line_width=line_width,
            line_dash=line_dash,
            alpha=alpha,
            legend_label=name if name and not params.get('no_legend') else None
            # line_shape is not a valid Bokeh parameter, removed.
        )

        # Handle categorical y-axis mapping if provided
        y_axis_mapping = params.get('y_axis_mapping')
        if y_axis_mapping:
            from bokeh.models import FixedTicker, CustomJSTickFormatter # Import CustomJSTickFormatter
            num_to_category = y_axis_mapping
            ticks = sorted(list(num_to_category.keys())) # Ensure ticks are sorted
            overrides = {i: label for i, label in num_to_category.items()}
            figure.yaxis.ticker = FixedTicker(ticks=ticks)
            figure.yaxis.major_label_overrides = overrides

            # Add category names to source for hover tool
            category_names = [num_to_category.get(val, 'Unknown') for val in source.data['y']]
            source.add(category_names, 'category_name')

            # Adjust hover tool for categorical data
            hover_tooltips = [
                ('Time', '@x{%F %T}'),
                ('Stage', '@category_name') # Use the added category name column
            ]
            hover_formatters = {'@x': 'datetime'}
        else:
            # Default hover tooltips for numerical data
            hover_tooltips = [
                ('Time', '@x{%F %T}'),
                ('Value', '@y')
            ]
            hover_formatters = {'@x': 'datetime'}

        # Add hover tooltip if requested
        if 'hover' in figure.tools and params.get('add_hover', True):
            hover = next((tool for tool in figure.tools if isinstance(tool, HoverTool)), None)
            if hover is None:
                hover = HoverTool(
                    tooltips=hover_tooltips,
                    formatters=hover_formatters,
                    mode='vline' # Keep vline mode for time series
                )
                figure.add_tools(hover)
            else:
                # Update existing hover tool if needed
                hover.tooltips = hover_tooltips
                hover.formatters = hover_formatters
            
        return line
    
    def add_scatter_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """
        Add a scatter plot to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the plot to
            x: X-axis data
            y: Y-axis data
            **kwargs: Optional styling parameters
                - fill_color: Color of the markers
                - size: Size of the markers
                - marker: Marker type ('circle', 'square', etc.)
                - alpha: Transparency (0 to 1)
                - name: Name for the legend
                
        Returns:
            The Bokeh scatter renderer object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract styling parameters
        fill_color = params.get('fill_color', params.get('color', 'blue'))
        size = params.get('size', 8)
        marker = params.get('marker', 'circle')
        alpha = params.get('alpha', 0.7)
        name = params.get('name', None)
        
        # Create a ColumnDataSource
        source_data = {'x': x, 'y': y}
        if isinstance(x, pd.Series):
            source_data['x'] = x.values
        if isinstance(y, pd.Series):
            source_data['y'] = y.values
            
        # Handle NaN values
        valid_mask = ~(np.isnan(source_data['x']) | np.isnan(source_data['y']))
        if not valid_mask.all():
            source_data['x'] = source_data['x'][valid_mask]
            source_data['y'] = source_data['y'][valid_mask]
            
        source = ColumnDataSource(data=source_data)
        
        # Add the scatter plot based on marker type
        if marker == 'circle':
            scatter = figure.circle(
                'x', 'y', 
                source=source,
                fill_color=fill_color, 
                size=size,
                alpha=alpha,
                legend_label=name if name and not params.get('no_legend') else None
            )
        elif marker == 'square':
            scatter = figure.square(
                'x', 'y', 
                source=source,
                fill_color=fill_color, 
                size=size,
                alpha=alpha,
                legend_label=name if name and not params.get('no_legend') else None
            )
        else:  # Default to circle
            scatter = figure.circle(
                'x', 'y', 
                source=source,
                fill_color=fill_color, 
                size=size,
                alpha=alpha,
                legend_label=name if name and not params.get('no_legend') else None
            )
        
        # Add hover tooltip if requested
        if 'hover' in figure.tools and params.get('add_hover', True):
            tooltips = params.get('tooltips', [
                ('X', '@x'),
                ('Y', '@y')
            ])
            
            hover = next((tool for tool in figure.tools if isinstance(tool, HoverTool)), None)
            if hover is None:
                hover = HoverTool(
                    tooltips=tooltips,
                    mode='mouse'
                )
                figure.add_tools(hover)
            
        return scatter
    
    def add_heatmap(self, figure: Any, data: Any, **kwargs) -> Any:
        """
        Add a heatmap to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the heatmap to
            data: 2D data array (numpy array or pandas DataFrame)
            **kwargs: Optional styling parameters
                - palette: Color palette to use
                - x_range: Range of x-coordinates
                - y_range: Range of y-coordinates
                - alpha: Transparency (0 to 1)
                
        Returns:
            The Bokeh image renderer object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract styling parameters
        palette = params.get('palette', 'Viridis256')
        alpha = params.get('alpha', 1.0)
        
        # Prepare the data
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Handle time-indexed data
        x_range = params.get('x_range', [0, data.shape[1]])
        y_range = params.get('y_range', [0, data.shape[0]])
        
        # Special case for time domain spectrograms
        if params.get('is_spectrogram', False) and params.get('x_axis_type') == 'datetime':
            if 'x_axis_data' in params:
                # For spectrograms with time on x-axis
                x_data = params['x_axis_data']
                x_range = [x_data[0], x_data[-1]]
                
        # Add the heatmap
        image = figure.image(
            image=[data],
            x=x_range[0],
            y=y_range[0],
            dw=x_range[1] - x_range[0],
            dh=y_range[1] - y_range[0],
            palette=palette,
            alpha=alpha
        )
        
        # Add color bar if requested
        if params.get('add_colorbar', True):
            from bokeh.models import ColorBar, LinearColorMapper
            
            # Create color mapper
            color_mapper = LinearColorMapper(palette=palette, low=data.min(), high=data.max())
            
            # Create color bar
            color_bar = ColorBar(
                color_mapper=color_mapper,
                location=(0, 0),
                title=params.get('colorbar_title', '')
            )
            
            figure.add_layout(color_bar, 'right')
        
        return image
    
    def create_grid_layout(self, figures: List[Any], rows: int, cols: int, **kwargs) -> Any:
        """
        Create a grid layout of multiple Bokeh figures.
        
        Args:
            figures: List of Bokeh figures to arrange in a grid
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            **kwargs: Optional layout parameters
                - toolbar_location: Location of the toolbar ('above', 'below', 'left', 'right')
                - sizing_mode: How the layout responds to window resizing
                - link_x_axes: Whether to link x-axes across all figures (default: True)
                - subplot_titles: List of titles for each subplot
                
        Returns:
            A Bokeh gridplot object containing the arranged figures
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract layout parameters
        toolbar_location = params.get('toolbar_location', 'right')
        sizing_mode = params.get('sizing_mode', 'stretch_both')  # Default to responsive
        
        # Apply subplot titles if available
        subplot_titles = params.get('subplot_titles', [])
        if subplot_titles and len(subplot_titles) == len(figures):
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Applying {len(subplot_titles)} subplot titles to figures")
            
            for i, fig in enumerate(figures):
                if i < len(subplot_titles):
                    fig.title.text = subplot_titles[i]
        
        # Get linking parameter, default to True
        link_x_axes = params.get('link_x_axes', True)
        if link_x_axes and len(figures) > 1:
            # Link x-ranges across all plots
            x_range = figures[0].x_range
            for fig in figures[1:]:
                fig.x_range = x_range
        
        # Organize figures into a 2D list
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(figures):
                    row.append(figures[idx])
                else:
                    # Create a blank figure for empty spots
                    blank = figure(
                        width=params.get('width', 800),
                        height=params.get('height', 400),
                        tools=''
                    )
                    blank.grid.visible = False
                    blank.axis.visible = False
                    row.append(blank)
            grid.append(row)
            
        # Create the grid layout
        return gridplot(
            grid,
            toolbar_location=toolbar_location,
            sizing_mode=sizing_mode
        )
    
    def add_legend(self, figure: Any, **kwargs) -> None:
        """
        Add or customize the legend of a Bokeh figure or layout.
        
        Args:
            figure: The Bokeh figure or layout object.
            **kwargs: Optional legend parameters
                - location: Legend position ('top_left', 'top_right', etc.)
                - orientation: Legend orientation ('horizontal', 'vertical')
                - click_policy: How clicking on legend items affects the plot
                - background_legend_items: Dict mapping category names to colors for background legend
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        logger = logging.getLogger(__name__)
        
        try:
            # Handle background legend items if provided
            background_items = params.get('background_legend_items')
    
            # Recursively find the first figure to attach the legend
            def _find_fig(obj):
                if hasattr(obj, 'add_layout'):
                    return obj
                # Sequence types
                if isinstance(obj, (list, tuple)):
                    for it in obj:
                        found = _find_fig(it)
                        if found:
                            return found
                # Layout with children
                children = getattr(obj, 'children', None)
                if children:
                    for child in children:
                        found = _find_fig(child)
                        if found:
                            return found
                return None
            actual_target_figure = _find_fig(figure)
            if not actual_target_figure:
                logger.warning("Could not find any figure in layout to add legend")
                return
    
            # --- Add Background Legend (if requested and target found) ---
            if background_items:
                logger.debug(f"Adding background legend items: {list(background_items.keys())}")
                legend_items = []
                alpha = params.get('alpha', 0.3)
                
                # Use the exact same default colors as in visualize_hypnogram to ensure consistency
                default_colors = {
                    'Awake': '#FF6347',  # Tomato Red
                    'N1': '#ADD8E6',     # Light Blue
                    'N2': '#4169E1',     # Royal Blue
                    'N3': '#00008B',     # Dark Blue
                    'REM': '#9400D3',    # Dark Violet
                    'Unknown': '#A9A9A9'  # Dark Gray
                }
                
                # If we have stage_colors from kwargs, use those instead (passed from visualize_collection)
                stage_colors = params.get('stage_colors', default_colors)
                
                # Use the same alpha as in visualize_hypnogram for consistency
                # In visualize_hypnogram, alpha defaults to 1.0 (line 1037)
                # We must match this value exactly to ensure visual consistency
                plot_alpha = params.get('alpha', 1.0)  # Match default in visualize_hypnogram
    
                try:
                    for stage, color in background_items.items():
                        # Always use the canonical stage colors if available, otherwise fall back to provided color
                        actual_color = stage_colors.get(stage, color)
                        
                        # Create an invisible dummy renderer on the target figure
                        # This renderer exists solely to provide an item for the legend
                        dummy_renderer = actual_target_figure.rect(
                            x=0, y=0,  # Position doesn't matter for dummy renderer
                            width=10, height=10,  # Large enough to be visible in legend
                            fill_color=actual_color,
                            line_color=actual_color,
                            fill_alpha=plot_alpha,  # Use the same alpha as in the plot
                            line_alpha=1.0,  # Keep border fully visible
                            visible=True  # Make sure it's visible
                        )
                        # Create a LegendItem linking the label without 'Stage' suffix
                        legend_items.append(LegendItem(label=stage, renderers=[dummy_renderer]))
    
                    # Clear existing legends to avoid duplicates
                    if hasattr(actual_target_figure, 'legend') and actual_target_figure.legend:
                        actual_target_figure.legend[:] = []
    
                    # Create and add the new legend
                    legend = Legend(
                        items=legend_items,
                        location=params.get('legend_location', 'center'),
                        orientation=params.get('legend_orientation', 'horizontal'),
                        click_policy=params.get('legend_click_policy', 'hide'),
                        label_text_font_size=params.get('legend_font_size', '9pt'),
                        border_line_alpha=params.get('legend_border_alpha', 0.3),
                        background_fill_alpha=params.get('legend_bg_alpha', 0.5),
                        visible=True
                    )
                    
                    # Add the legend to the target figure
                    actual_target_figure.add_layout(legend, params.get('legend_position', 'above'))
                    logger.debug(f"Successfully added background legend with {len(legend_items)} items")
                except Exception as e:
                    logger.error(f"Error creating background legend items: {str(e)}")
    
            # --- Handle Standard Legends (if no background items requested) ---
            elif not background_items:
                # Check if the target figure has renderers with legend labels
                has_legend_items = False
                if hasattr(actual_target_figure, 'renderers'):
                    has_legend_items = any(
                        hasattr(r, 'legend_label') and r.legend_label and getattr(r, 'visible', True)
                        for r in actual_target_figure.renderers
                    )
    
                if has_legend_items:
                    # Configure existing legend or add a new one
                    if (hasattr(actual_target_figure, 'legend') and 
                        isinstance(actual_target_figure.legend, list) and 
                        actual_target_figure.legend):
                        legend_obj = actual_target_figure.legend[0]
                    else:
                        # Add a new legend if none exists
                        actual_target_figure.add_layout(Legend(), params.get('legend_position', 'above'))
                        legend_obj = actual_target_figure.legend[0] if actual_target_figure.legend else None
    
                    if legend_obj:
                        legend_obj.location = params.get('legend_location', 'center')
                        legend_obj.orientation = params.get('legend_orientation', 'horizontal')
                        legend_obj.click_policy = params.get('legend_click_policy', 'hide')
                        legend_obj.label_text_font_size = params.get('legend_font_size', '10pt')
                        legend_obj.border_line_alpha = params.get('legend_border_alpha', 0.3)
                        legend_obj.background_fill_alpha = params.get('legend_bg_alpha', 0.5)
                        legend_obj.visible = True
                else:
                    # Hide legend if no items to show
                    if (hasattr(actual_target_figure, 'legend') and 
                        isinstance(actual_target_figure.legend, list) and 
                        actual_target_figure.legend):
                        actual_target_figure.legend[0].visible = False
        
        except Exception as e:
            logger.error(f"Error in add_legend: {str(e)}", exc_info=True)
            # Fail gracefully - don't crash the visualization process
    
    def set_title(self, figure: Any, title: str, **kwargs) -> None:
        """
        Set the title of a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to set the title for
            title: The title text
            **kwargs: Optional title styling parameters
                - text_font_size: Font size
                - text_font_style: Font style ('normal', 'italic', 'bold')
                - align: Text alignment ('left', 'center', 'right')
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract title parameters
        font_size = params.get('title_font_size', '14pt')
        font_style = params.get('title_font_style', 'bold')
        align = params.get('title_align', 'center')
        
        # Set the title
        figure.title.text = title
        figure.title.text_font_size = font_size
        figure.title.text_font_style = font_style
        figure.title.align = align
    
    def set_axis_labels(self, figure: Any, x_label: Optional[str] = None, 
                        y_label: Optional[str] = None, **kwargs) -> None:
        """
        Set axis labels for a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to set axis labels for
            x_label: Optional label for the x-axis
            y_label: Optional label for the y-axis
            **kwargs: Optional label styling parameters
                - font_size: Font size for axis labels
                - font_style: Font style for axis labels
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract label parameters
        font_size = params.get('axis_label_font_size', '9pt') # Further reduced default size
        font_style = params.get('axis_label_font_style', 'normal')
        
        # Set x-axis label if provided
        if x_label is not None:
            figure.xaxis.axis_label = x_label
            figure.xaxis.axis_label_text_font_size = font_size
            figure.xaxis.axis_label_text_font_style = font_style
            
        # Set y-axis label if provided
        if y_label is not None:
            figure.yaxis.axis_label = y_label
            figure.yaxis.axis_label_text_font_size = font_size
            figure.yaxis.axis_label_text_font_style = font_style
    
    def set_axis_limits(self, figure: Any, x_min: Optional[Any] = None, 
                        x_max: Optional[Any] = None, y_min: Optional[Any] = None, 
                        y_max: Optional[Any] = None) -> None:
        """
        Set axis limits for a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to set axis limits for
            x_min: Optional minimum value for the x-axis
            x_max: Optional maximum value for the x-axis
            y_min: Optional minimum value for the y-axis
            y_max: Optional maximum value for the y-axis
        """
        # Set x-axis range if provided
        if x_min is not None or x_max is not None:
            current_x_range = figure.x_range.bounds or (None, None)
            new_x_min = x_min if x_min is not None else current_x_range[0]
            new_x_max = x_max if x_max is not None else current_x_range[1]
            figure.x_range.bounds = (new_x_min, new_x_max)
            
        # Set y-axis range if provided
        if y_min is not None or y_max is not None:
            current_y_range = figure.y_range.bounds or (None, None)
            new_y_min = y_min if y_min is not None else current_y_range[0]
            new_y_max = y_max if y_max is not None else current_y_range[1]
            figure.y_range.bounds = (new_y_min, new_y_max)
    
    def add_hover_tooltip(self, figure: Any, tooltips: Any, **kwargs) -> None:
        """
        Add hover tooltips to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add tooltips to
            tooltips: List of (name, value) tuples for tooltips
            **kwargs: Optional tooltip parameters
                - mode: Hover mode ('mouse', 'vline', 'hline')
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract tooltip parameters
        mode = params.get('hover_mode', 'mouse')
        
        # Create and add hover tool
        hover = HoverTool(
            tooltips=tooltips,
            mode=mode
        )
        
        # If formatters are provided, apply them
        if 'hover_formatters' in params:
            hover.formatters = params['hover_formatters']
            
        figure.add_tools(hover)
    
    def add_vertical_line(self, figure: Any, x: Any, **kwargs) -> Any:
        """
        Add a vertical line to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the line to
            x: The x-coordinate of the line
            **kwargs: Optional line styling parameters
                - color: Line color
                - line_width: Line width
                - line_dash: Line dash pattern
                - alpha: Line transparency
                
        Returns:
            The Bokeh Span object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract line parameters
        color = params.get('line_color', params.get('color', 'red'))
        line_width = params.get('line_width', 1)
        line_dash = params.get('line_dash', 'solid')
        alpha = params.get('alpha', 0.7)
        
        # Create vertical line span
        vline = Span(
            location=x,
            dimension='height',
            line_color=color,
            line_width=line_width,
            line_dash=line_dash,
            line_alpha=alpha
        )
        
        figure.add_layout(vline)
        return vline
    
    def add_horizontal_line(self, figure: Any, y: Any, **kwargs) -> Any:
        """
        Add a horizontal line to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the line to
            y: The y-coordinate of the line
            **kwargs: Optional line styling parameters
                - color: Line color
                - line_width: Line width
                - line_dash: Line dash pattern
                - alpha: Line transparency
                
        Returns:
            The Bokeh Span object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract line parameters
        color = params.get('line_color', params.get('color', 'green'))
        line_width = params.get('line_width', 1)
        line_dash = params.get('line_dash', 'solid')
        alpha = params.get('alpha', 0.7)
        
        # Create horizontal line span
        hline = Span(
            location=y,
            dimension='width',
            line_color=color,
            line_width=line_width,
            line_dash=line_dash,
            line_alpha=alpha
        )
        
        figure.add_layout(hline)
        return hline

    # Removed _get_sleep_stage_regions, now using static method from VisualizerBase

    def add_region(self, figure: Any, x_start: Any, x_end: Any, **kwargs) -> Any:
        """
        Add a highlighted region to a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the region to
            x_start: The starting x-coordinate of the region
            x_end: The ending x-coordinate of the region
            **kwargs: Optional region styling parameters
                - color: Fill color
                - alpha: Fill transparency
                - hatch_pattern: Hatch pattern (None, '/', '\\', etc.)
                
        Returns:
            The Bokeh BoxAnnotation object
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract region parameters
        color = params.get('fill_color', params.get('color', 'yellow'))
        alpha = params.get('alpha', 0.2)
        hatch_pattern = params.get('hatch_pattern', None)
        
        # Create box annotation
        region = BoxAnnotation(
            left=x_start,
            right=x_end,
            fill_color=color,
            fill_alpha=alpha,
            hatch_pattern=hatch_pattern
        )
        
        figure.add_layout(region)
        return region
        
    def add_categorical_regions(self, figure: Any, start_times: Any, end_times: Any, 
                                categories: Any, category_map: Dict[str, str],
                                **kwargs) -> List[Any]:
        """
        Add colored background regions for categorical data using BoxAnnotation.

        Args:
            figure: The Bokeh figure to add the regions to.
            start_times: Series/array of start times for each interval.
            end_times: Series/array of end times for each interval.
            categories: Series/array of category labels for each interval.
            category_map: Dictionary mapping category labels to colors.
            **kwargs: Optional styling parameters (e.g., alpha).

        Returns:
            List of BoxAnnotation objects added.
        """
        regions = []
        params = {**self.default_params, **kwargs}
        alpha = params.get('alpha', 0.2) # Make background more subtle by default

        # Add a BoxAnnotation for each interval
        for start, end, cat in zip(start_times, end_times, categories):
            if pd.isna(start) or pd.isna(end) or start == end: # Skip invalid or zero-duration intervals
                continue

            color = category_map.get(cat, 'gray') # Default to gray

            region = BoxAnnotation(
                # bottom=0, top=1, # Spans the full default y-range of the plot area
                # y_range_name='default', # Ensure it uses the primary y-axis range
                left=start,
                right=end,
                fill_color=color,
                fill_alpha=alpha,
                level='underlay' # Crucial: Draw behind data lines
            )
            figure.add_layout(region)
            regions.append(region)

        # Legend for background colors is handled separately in visualize_collection
        # as it needs to be added only once to the overall layout.

        return regions

    def save(self, figure: Any, filename: str, format: str = "html", **kwargs) -> None:
        """
        Save a Bokeh figure to a file.
        
        Args:
            figure: The Bokeh figure to save
            filename: The output filename
            format: The output format (html, png, svg)
            **kwargs: Optional export parameters
                - title: Page title for HTML output
                - resources: Resource handling ('cdn', 'inline')
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Handle different output formats
        if format.lower() in ('html', 'htm'):
            # Extract HTML parameters
            title = params.get('title', 'Bokeh Plot')
            resources = params.get('resources', 'cdn')
            
            # Function to add placeholder for empty plots
            def add_placeholder(fig):
                if hasattr(fig, 'renderers') and not fig.renderers:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Figure has no renderers, adding a placeholder message")
                    from bokeh.models import Label
                    fig.add_layout(Label(x=0, y=0, text="No data to display", text_font_size="20pt", text_color="red"))
            
            # Handle both single figures and grid layouts
            import logging
            logger = logging.getLogger(__name__)
            
            if isinstance(figure, GridBox):
                # For grid layouts, check each child figure
                logger.debug("Checking gridplot children for empty renderers")
                for child in figure.children:
                    if isinstance(child, tuple):  # (figure, row, col) format
                        add_placeholder(child[0])
                    else:
                        add_placeholder(child)
            else:
                # For single figure
                add_placeholder(figure)
            
            # Update figure dimensions if specified
            width = params.get('width')
            height = params.get('height')
            
            # Always log the dimensions we're applying
            
            if width is not None:
                figure.width = width
                logger.debug(f"Setting Bokeh figure width to {width}")
            if height is not None:
                figure.height = height
                logger.debug(f"Setting Bokeh figure height to {height}")
                
            # Set dimensions if provided
            if width is not None:
                figure.width = width
                logger.debug(f"Setting Bokeh figure width to {width}")
            if height is not None:
                figure.height = height
                logger.debug(f"Setting Bokeh figure height to {height}")
        
            # Log the current sizing mode 
            logger.debug(f"Using {figure.sizing_mode} sizing")
            
            # Configure output and save
            output_file(filename, title=title, mode='inline' if resources == 'inline' else 'cdn')
            
            from bokeh.io import save
            save(figure)
            
        elif format.lower() in ('png', 'jpg', 'jpeg', 'svg', 'pdf'):
            # For static image formats, use export_png or export_svg from bokeh.io
            if format.lower() in ('png', 'jpg', 'jpeg'):
                try:
                    from bokeh.io import export_png
                    export_png(figure, filename=filename)
                except ImportError:
                    raise ImportError(
                        "Exporting to PNG requires selenium and phantomjs. "
                        "Install with 'pip install selenium phantomjs'."
                    )
            elif format.lower() == 'svg':
                try:
                    from bokeh.io import export_svg
                    export_svg(figure, filename=filename)
                except ImportError:
                    raise ImportError(
                        "Exporting to SVG requires svgwrite. "
                        "Install with 'pip install svgwrite'."
                    )
            elif format.lower() == 'pdf':
                try:
                    from bokeh.io import export_svg
                    import cairosvg
                    # Export to SVG first
                    svg_filename = filename.replace('.pdf', '.svg')
                    export_svg(figure, filename=svg_filename)
                    # Convert SVG to PDF
                    cairosvg.svg2pdf(file_obj=open(svg_filename, 'rb'), write_to=filename)
                    # Remove temporary SVG file
                    os.remove(svg_filename)
                except ImportError:
                    raise ImportError(
                        "Exporting to PDF requires svgwrite and cairosvg. "
                        "Install with 'pip install svgwrite cairosvg'."
                    )
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def show(self, figure: Any) -> None:
        """
        Display a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to display
        """
        show(figure)

    def visualize_hypnogram(self, figure: Any, signal: EEGSleepStageSignal, **kwargs) -> Any:
        """
        Create a sleep stage hypnogram visualization using Bokeh quads.
        """
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

        # Define stronger default colors for each stage
        default_colors = {
            'Awake': '#FF6347',  # Tomato Red
            'N1': '#ADD8E6',     # Light Blue
            'N2': '#4169E1',     # Royal Blue
            'N3': '#00008B',     # Dark Blue
            'REM': '#9400D3',    # Dark Violet
            'Unknown': '#A9A9A9' # Dark Gray
        }
        # Ensure default colors cover the default order (using a fallback palette if needed)
        palette = Category20[max(3, min(20, num_stages))] # Use a larger palette if needed
        for i, stage in enumerate(stage_order):
             if stage not in default_colors:
                 default_colors[stage] = palette[i % len(palette)]

        stage_colors = params.get('stage_colors', default_colors)

        # Prepare data for plotting quads
        quad_data = {
            'left': [], 'right': [], 'top': [], 'bottom': [],
            'color': [], 'alpha': [], 'stage_name': [], 'duration_min': []
        }
        # Use stronger defaults: opaque alpha, black lines
        alpha = params.get('alpha', 1.0) # Default to fully opaque
        line_color = params.get('line_color', "black") # Default to black lines
        line_width = params.get('line_width', 1)

        # Iterate through the dataframe to create segments for quads
        for i in range(len(df) - 1):
            stage = df[stage_column].iloc[i]
            # Map unexpected stages to 'Unknown'
            if stage not in stage_to_num:
                logger.debug(f"Mapping stage '{stage}' to 'Unknown'")
                stage = 'Unknown'

            stage_num = stage_to_num[stage]
            start_time = df.index[i]
            end_time = df.index[i+1]
            duration = (end_time - start_time).total_seconds() / 60.0

            quad_data['left'].append(start_time)
            quad_data['right'].append(end_time)
            quad_data['top'].append(stage_num + 0.4)  # Add slight padding
            quad_data['bottom'].append(-0.5)          # Extend to bottom
            quad_data['color'].append(stage_colors.get(stage, '#CCCCCC')) # Fallback color
            quad_data['alpha'].append(alpha)
            quad_data['stage_name'].append(stage)
            quad_data['duration_min'].append(duration)

        if not quad_data['left']:
             logger.warning("No valid sleep stage segments found to plot.")
             return None

        source = ColumnDataSource(data=quad_data)

        # Add quads to the figure
        quads = figure.quad(
            left='left', right='right', top='top', bottom='bottom',
            color='color', alpha='alpha', source=source,
            line_color=line_color, line_width=line_width,
            name='hypnogram_quads' # Give the renderer a name
        )

        # Customize y-axis to show stage names
        figure.yaxis.ticker = [i for i in range(num_stages)]
        figure.yaxis.major_label_overrides = {i: stage for i, stage in enumerate(stage_order)}
        figure.y_range = Range1d(-0.5, num_stages - 0.5) # Set y-range explicitly

        # Add hover tool for better interactivity
        if params.get('add_hover', True) and 'hover' in figure.tools:
            hover = next((tool for tool in figure.tools if isinstance(tool, HoverTool)), None)
            if hover:
                hover.tooltips = [
                    ('Time', '@left{%F %T}'),
                    ('Stage', '@stage_name'),
                    ('Duration', '@duration_min{0.1f} min')
                ]
                hover.formatters = {'@left': 'datetime'}
                hover.mode = 'vline'
                hover.renderers = [quads] # Attach hover tool specifically to quads

        return quads # Return the quad renderer

    def _add_statistics_annotation(self, figure: Any, stats_text: List[str], **kwargs) -> None:
        """Add sleep statistics annotation using Bokeh Label."""
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}

        # Extract annotation parameters
        x = params.get('stats_x', 10)
        y = params.get('stats_y', 10)
        x_units = params.get('stats_x_units', 'screen')
        y_units = params.get('stats_y_units', 'screen')
        text_font_size = params.get('stats_font_size', '8pt') # Smaller default
        bg_color = params.get('stats_bg_color', 'white')
        bg_alpha = params.get('stats_bg_alpha', 0.7)
        border_color = params.get('stats_border_color', 'lightgrey')
        border_alpha = params.get('stats_border_alpha', 0.5)

        # Join the list of strings into a single multi-line string
        annotation_text = "\n".join(stats_text)

        # Create and add the label
        stats_label = Label(
            x=x, y=y,
            x_units=x_units, y_units=y_units,
            text=annotation_text,
            text_font_size=text_font_size,
            background_fill_color=bg_color,
            background_fill_alpha=bg_alpha,
            border_line_color=border_color,
            border_line_alpha=border_alpha,
            border_line_width=1,
            text_align='left',
            level='annotation' # Ensure it's drawn on top
        )
        figure.add_layout(stats_label)
