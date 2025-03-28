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
        Span, BoxAnnotation, Legend, LegendItem, GridBox
    )
    from bokeh.layouts import gridplot
    from bokeh.palettes import Category10_10
except ImportError:
    raise ImportError(
        "Bokeh is required for BokehVisualizer. "
        "Install it with 'pip install bokeh'."
    )

from .base import VisualizerBase

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
        )
        
        # Add hover tooltip if requested
        if 'hover' in figure.tools and params.get('add_hover', True):
            tooltips = params.get('tooltips', [
                ('Time', '@x{%F %T}'),
                ('Value', '@y')
            ])
            
            hover = next((tool for tool in figure.tools if isinstance(tool, HoverTool)), None)
            if hover is None:
                hover = HoverTool(
                    tooltips=tooltips,
                    formatters={'@x': 'datetime'},
                    mode='vline'
                )
                figure.add_tools(hover)
            
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
        Add or customize the legend of a Bokeh figure.
        
        Args:
            figure: The Bokeh figure to add the legend to
            **kwargs: Optional legend parameters
                - location: Legend position ('top_left', 'top_right', etc.)
                - orientation: Legend orientation ('horizontal', 'vertical')
                - click_policy: How clicking on legend items affects the plot
        """
        # Combine default parameters with provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Extract legend parameters
        location = params.get('legend_location', 'top_right')
        orientation = params.get('legend_orientation', 'vertical')
        click_policy = params.get('legend_click_policy', 'hide')
        
        # Configure the legend if it exists
        if hasattr(figure, 'legend') and figure.legend:
            figure.legend.location = location
            figure.legend.orientation = orientation
            figure.legend.click_policy = click_policy
            
            # Additional styling
            figure.legend.label_text_font_size = params.get('legend_font_size', '10pt')
            figure.legend.border_line_alpha = params.get('legend_border_alpha', 0.3)
            figure.legend.background_fill_alpha = params.get('legend_bg_alpha', 0.5)
    
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
        font_size = params.get('axis_label_font_size', '12pt')
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
