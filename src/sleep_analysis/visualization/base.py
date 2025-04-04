"""
Abstract base class for visualization backends.

This module defines the VisualizerBase abstract base class which provides
a unified interface for all visualization backends used in the sleep analysis framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
from abc import abstractmethod # Added import
from typing import Any, Dict, List, Optional # Added imports

from ..core.signal_collection import SignalCollection
from ..core.signal_data import SignalData
from ..signals.eeg_sleep_stage_signal import EEGSleepStageSignal # Added import
from ..signal_types import SignalType

class VisualizerBase(ABC):
    """
    Abstract base class defining the interface for visualization backends.
    
    This class defines a common interface for visualization libraries like Bokeh and Plotly,
    allowing for a consistent API regardless of the specific backend used.
    """
    
    def __init__(self, default_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualizer with default parameters.
        
        Args:
            default_params: Optional dictionary of default visualization parameters
        """
        self.default_params = default_params or {}
    
    @abstractmethod
    def create_figure(self, **kwargs) -> Any:
        """
        Create a new figure/plot container.
        
        Args:
            **kwargs: Optional parameters for figure creation
        
        Returns:
            A backend-specific figure object
        """
        pass
    
    @abstractmethod
    def add_line_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """
        Add a line plot to an existing figure.
        
        Args:
            figure: The figure to add the plot to
            x: X-axis data
            y: Y-axis data
            **kwargs: Optional styling parameters
            
        Returns:
            The plot element that was added (e.g., a backend-specific line object)
        """
        pass
    
    @abstractmethod
    def add_scatter_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """
        Add a scatter plot to an existing figure.
        
        Args:
            figure: The figure to add the plot to
            x: X-axis data
            y: Y-axis data
            **kwargs: Optional styling parameters
            
        Returns:
            The plot element that was added (e.g., a backend-specific scatter object)
        """
        pass
    
    @abstractmethod
    def add_heatmap(self, figure: Any, data: Any, **kwargs) -> Any:
        """
        Add a heatmap to an existing figure.
        
        Args:
            figure: The figure to add the heatmap to
            data: 2D data array
            **kwargs: Optional styling parameters
            
        Returns:
            The plot element that was added (e.g., a backend-specific heatmap object)
        """
        pass
    
    @abstractmethod
    def create_grid_layout(self, figures: List[Any], rows: int, cols: int, **kwargs) -> Any:
        """
        Create a grid layout of multiple figures.
        
        Args:
            figures: List of figures to arrange in a grid
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            **kwargs: Optional layout parameters
            
        Returns:
            A backend-specific layout object containing the arranged figures
        """
        pass
    
    @abstractmethod
    def add_legend(self, figure: Any, **kwargs) -> None:
        """
        Add a legend to a figure.
        
        Args:
            figure: The figure to add the legend to
            **kwargs: Optional legend parameters
        """
        pass
    
    @abstractmethod
    def set_title(self, figure: Any, title: str, **kwargs) -> None:
        """
        Set the title of a figure.
        
        Args:
            figure: The figure to set the title for
            title: The title text
            **kwargs: Optional title styling parameters
        """
        pass
    
    @abstractmethod
    def set_axis_labels(self, figure: Any, x_label: Optional[str] = None, 
                        y_label: Optional[str] = None, **kwargs) -> None:
        """
        Set axis labels for a figure.
        
        Args:
            figure: The figure to set axis labels for
            x_label: Optional label for the x-axis
            y_label: Optional label for the y-axis
            **kwargs: Optional label styling parameters
        """
        pass
    
    @abstractmethod
    def set_axis_limits(self, figure: Any, x_min: Optional[Any] = None, 
                        x_max: Optional[Any] = None, y_min: Optional[Any] = None, 
                        y_max: Optional[Any] = None) -> None:
        """
        Set axis limits for a figure.
        
        Args:
            figure: The figure to set axis limits for
            x_min: Optional minimum value for the x-axis
            x_max: Optional maximum value for the x-axis
            y_min: Optional minimum value for the y-axis
            y_max: Optional maximum value for the y-axis
        """
        pass
    
    @abstractmethod
    def add_hover_tooltip(self, figure: Any, tooltips: Any, **kwargs) -> None:
        """
        Add hover tooltips to a figure.
        
        Args:
            figure: The figure to add tooltips to
            tooltips: The tooltip configuration (backend-specific)
            **kwargs: Optional tooltip parameters
        """
        pass
    
    @abstractmethod
    def add_vertical_line(self, figure: Any, x: Any, **kwargs) -> Any:
        """
        Add a vertical line to a figure.
        
        Args:
            figure: The figure to add the line to
            x: The x-coordinate of the line
            **kwargs: Optional line styling parameters
            
        Returns:
            The line object that was added
        """
        pass
    
    @abstractmethod
    def add_horizontal_line(self, figure: Any, y: Any, **kwargs) -> Any:
        """
        Add a horizontal line to a figure.
        
        Args:
            figure: The figure to add the line to
            y: The y-coordinate of the line
            **kwargs: Optional line styling parameters
            
        Returns:
            The line object that was added
        """
        pass
    
    @abstractmethod
    def add_categorical_regions(self, figure: Any, start_times: Any, end_times: Any, 
                                categories: Any, category_map: Dict[str, str], 
                                **kwargs) -> List[Any]:
        """
        Add colored regions representing categorical data over time intervals.
        
        Args:
            figure: The figure to add the regions to
            start_times: Series/array of start times for each interval
            end_times: Series/array of end times for each interval
            categories: Series/array of category labels for each interval
            category_map: Dictionary mapping category labels to colors
            **kwargs: Optional styling parameters (e.g., opacity, y_range)
            
        Returns:
            List of region objects that were added
        """
        pass
        
    @abstractmethod
    def add_region(self, figure: Any, x_start: Any, x_end: Any, **kwargs) -> Any:
        """
        Add a highlighted region (typically for annotations) to a figure.
        
        Args:
            figure: The figure to add the region to
            x_start: The starting x-coordinate of the region
            x_end: The ending x-coordinate of the region
            **kwargs: Optional region styling parameters
            
        Returns:
            The region object that was added
        """
        pass
    
    @abstractmethod
    def save(self, figure: Any, filename: str, format: str = "html", **kwargs) -> None:
        """
        Save a figure to a file.
        
        Args:
            figure: The figure to save
            filename: The output filename
            format: The output format (e.g., "html", "png", "svg")
            **kwargs: Optional export parameters
        """
        pass
    
    @abstractmethod
    def show(self, figure: Any) -> None:
        """
        Display a figure.
        
        Args:
            figure: The figure to display
        """
        pass

    @abstractmethod
    def visualize_hypnogram(self, figure: Any, signal: EEGSleepStageSignal, **kwargs) -> Any:
        """
        Create a sleep stage hypnogram visualization.

        Args:
            figure: The figure to add the hypnogram to
            signal: The EEGSleepStageSignal object containing sleep stage data
            **kwargs: Optional styling parameters

        Returns:
            Any visualization elements created
        """
        pass

    @abstractmethod
    def _add_statistics_annotation(self, figure: Any, stats_text: List[str], **kwargs) -> None:
        """
        Add statistics text annotation to a figure (backend-specific implementation).

        Args:
            figure: The figure to add the annotation to.
            stats_text: A list of strings, where each string is a line of statistics.
            **kwargs: Optional parameters for annotation styling and positioning.
        """
        pass

    # --- High-Level Plot Creation Methods ---

    def create_hypnogram_plot(self, signal: EEGSleepStageSignal, **kwargs) -> Any:
        """
        Create a hypnogram plot from an EEGSleepStageSignal.

        Args:
            signal: The EEGSleepStageSignal object containing sleep stage data
            **kwargs: Optional visualization parameters
                - title: Plot title
                - width: Plot width
                - height: Plot height
                - stage_colors: Dict mapping stages to colors
                - stage_order: List defining the order of stages on the y-axis
                - add_statistics: Whether to add sleep statistics (default: True)

        Returns:
            A figure object with the hypnogram
        """
        # Create figure
        fig = self.create_figure(
            title=kwargs.get('title', 'Sleep Stages Hypnogram'),
            width=kwargs.get('width', self.default_params.get('width')),
            height=kwargs.get('height', self.default_params.get('height')),
            x_axis_type='datetime'
        )

        # Add the hypnogram visualization
        self.visualize_hypnogram(fig, signal, **kwargs)

        # Set axis labels
        self.set_axis_labels(
            fig,
            x_label=kwargs.get('x_label', 'Time'),
            y_label=kwargs.get('y_label', 'Sleep Stage')
        )

        # Add sleep statistics if requested
        if kwargs.get('add_statistics', True):
            self._add_sleep_statistics(fig, signal, **kwargs)

        return fig

    def _add_sleep_statistics(self, figure: Any, signal: EEGSleepStageSignal, **kwargs) -> None:
        """
        Calculate and add sleep statistics to a hypnogram plot.

        Args:
            figure: The figure to add statistics to
            signal: The EEGSleepStageSignal object containing sleep stage data
            **kwargs: Optional parameters for statistics display
        """
        # Calculate statistics
        stage_distribution = signal.get_stage_distribution()
        if stage_distribution is None or stage_distribution.empty:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No stage distribution data found, skipping statistics annotation.")
            return

        # Format statistics text
        total_minutes = stage_distribution.sum()
        stats_text = [f"Total recording time: {total_minutes:.1f} minutes"]

        # Use the stage order from kwargs or default for consistent ordering
        stage_order = kwargs.get('stage_order', ['Awake', 'REM', 'N1', 'N2', 'N3', 'Unknown'])
        for stage in stage_order:
            if stage in stage_distribution.index:
                count = stage_distribution[stage]
                percentage = (count / total_minutes * 100).round(1) if total_minutes > 0 else 0
                stats_text.append(f"{stage}: {count:.1f} minutes ({percentage}%)")
            # Optionally add entries for stages present in data but not in order
            # Or handle 'Unknown' specifically if needed

        # Add statistics based on backend type (implementation in concrete classes)
        self._add_statistics_annotation(figure, stats_text, **kwargs)

    def create_time_series_plot(self, signal: SignalData, **kwargs) -> Any:
        """
        Create a time-series plot for a single signal.
        
        Args:
            signal: The signal to visualize
            **kwargs: Optional visualization parameters
            
        Returns:
            A backend-specific figure object
        """
        import logging # Import logging
        logger = logging.getLogger(__name__) # Get logger instance

        # Create figure
        figure = self.create_figure(**kwargs)
        
        # Get data
        data = signal.get_data()
        
        # Set appropriate x-axis type for datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            kwargs['x_axis_type'] = 'datetime'
            
        # Check if the primary data column is categorical
        is_categorical = False
        primary_col = data.columns[0] # Assume first column is primary
        if pd.api.types.is_categorical_dtype(data[primary_col]) or \
           pd.api.types.is_object_dtype(data[primary_col]) or \
           (hasattr(signal, 'signal_type') and signal.signal_type == SignalType.EEG_SLEEP_STAGE):
            is_categorical = True
            
        # Apply downsampling if specified in kwargs (only for numerical)
        if not is_categorical and 'downsample' in kwargs:
            downsample_param = kwargs['downsample']
            original_len = len(data) # Log original length
            logger.debug(f"Signal '{signal.metadata.name or 'Unnamed'}': Checking downsample ({downsample_param}). Original points: {original_len}") # Removed (Col: {col})

            if isinstance(downsample_param, int) and downsample_param > 0:
                # Interpret as max_points
                max_points = downsample_param
                if original_len > max_points:
                    import numpy as np
                    step = max(1, original_len // max_points)
                    data = data.iloc[::step]
                    logger.info(f"Signal '{signal.metadata.name or 'Unnamed'}': Downsampled from {original_len} to {len(data)} points (max: {max_points})") # Removed (Col: {col})
                else:
                    logger.debug(f"Signal '{signal.metadata.name or 'Unnamed'}': No downsampling needed ({original_len} <= {max_points})") # Removed (Col: {col})
            elif isinstance(downsample_param, float) and 0 < downsample_param < 1:
                 # Interpret as fraction (not implemented yet, could use resample)
                 pass # Placeholder for fractional downsampling
            elif isinstance(downsample_param, str):
                 # Interpret as time frequency (e.g., '1S', '1T')
                 try:
                     data = data.resample(downsample_param).mean() # Or first(), median() etc.
                     logger.info(f"Signal '{signal.metadata.name or 'Unnamed'}': Resampled from {original_len} to {len(data)} points using frequency '{downsample_param}'") # Removed (Col: {col})
                 except Exception as e:
                     # Logger already defined above, no need to import logging here
                     logger.warning(f"Could not apply time-based downsampling '{downsample_param}': {e}")

        # Get color palette if available
        palette = kwargs.get('palette', self.default_params.get('palette', None))
            
        # Loop through columns, assigning a different color to each
        for i, col in enumerate(data.columns):
            # Use signal metadata for line name if available
            line_name = kwargs.get('name', signal.metadata.name)
            
            if col != 'value' and line_name:
                # Add column name to the display name if there are multiple columns
                display_name = f"{line_name} - {col}" if len(data.columns) > 1 else line_name
            else:
                display_name = line_name
            
            # Create plot-specific kwargs that include the name but don't duplicate parameters
            plot_kwargs = kwargs.copy()
            plot_kwargs['name'] = display_name
            
            # Assign a different color for each column if palette is available
            if palette and len(data.columns) > 1:
                # For Bokeh palettes (lists/tuples)
                if hasattr(palette, '__getitem__') and isinstance(palette, (list, tuple)):
                    color_idx = i % len(palette)
                    plot_kwargs['color'] = palette[color_idx]
                else:
                    # For Plotly named palettes or any other case
                    plot_kwargs['color_index'] = i
            
            # Plot numerical data as lines
            if not is_categorical:
                self.add_line_plot(figure, data.index, data[col], **plot_kwargs)
            else:
                # Plot categorical data as a stepped line
                # 1. Define mapping from category to number
                #    Sort categories for consistent mapping
                unique_categories = sorted(data[col].unique())
                category_to_num = {cat: i for i, cat in enumerate(unique_categories)}
                num_to_category = {i: cat for cat, i in category_to_num.items()}
                
                # 2. Convert data to numerical representation
                numerical_data = data[col].map(category_to_num)
                # Ensure dtype is float to handle potential NaNs correctly in plotting backend
                numerical_data = numerical_data.astype(float) 
                
                # 3. Add plot kwargs specific to categorical line plot
                cat_plot_kwargs = plot_kwargs.copy()
                cat_plot_kwargs['line_shape'] = 'hv' # Use stepped line
                cat_plot_kwargs['y_axis_mapping'] = num_to_category # Pass mapping for axis setup
                
                # 4. Plot the numerical data
                self.add_line_plot(figure, data.index, numerical_data, **cat_plot_kwargs)
                
                # 5. Adjust y-axis label for categorical data
                self.set_axis_labels(figure, y_label='Stage') # Set y-label to 'Stage'
                # Backend implementation will use the mapping to set ticks/labels

        # Add metadata as labels
        title = kwargs.get('title', signal.metadata.name or ('Sleep Stage Plot' if is_categorical else 'Time Series Plot'))
        self.set_title(figure, title)
        
        x_label = kwargs.get('x_label', 'Time')
        
        # Set y-label only if not categorical (handled above for categorical)
        if not is_categorical:
            y_label = kwargs.get('y_label')
            if not y_label and hasattr(signal.metadata, 'units') and signal.metadata.units:
                try:
                    unit_name = signal.metadata.units.name if hasattr(signal.metadata.units, 'name') else str(signal.metadata.units)
                    y_label = f"{signal.metadata.name or 'Value'} ({unit_name})"
                except AttributeError: # Handle cases where units might not be an enum
                    y_label = f"{signal.metadata.name or 'Value'} ({signal.metadata.units})"
            elif not y_label:
                 y_label = signal.metadata.name or 'Value'
                 
            self.set_axis_labels(figure, x_label=x_label, y_label=y_label)
        else:
            # For categorical, only set x-label (y-label set above)
             self.set_axis_labels(figure, x_label=x_label)
        
        return figure
    
    def create_scatter_plot(self, x_signal: SignalData, y_signal: SignalData, **kwargs) -> Any:
        """
        Create a scatter plot comparing two signals.
        
        Args:
            x_signal: The signal for the x-axis
            y_signal: The signal for the y-axis
            **kwargs: Optional visualization parameters
            
        Returns:
            A backend-specific figure object
        """
        # Create figure
        figure = self.create_figure(**kwargs)
        
        # Get data and ensure alignment
        x_data = x_signal.get_data()
        y_data = y_signal.get_data()
        
        # Align the data using the index
        # This avoids issues with mismatched timestamps
        aligned_data = pd.merge(
            x_data, y_data, 
            left_index=True, right_index=True,
            how='inner',
            suffixes=('_x', '_y')
        )
        
        if aligned_data.empty:
            raise ValueError("No overlapping data points between the signals")
        
        x_col = aligned_data.columns[0]  # First column from x_data
        y_col = aligned_data.columns[-1]  # Last column from y_data
        
        # Create scatter plot
        self.add_scatter_plot(figure, aligned_data[x_col], aligned_data[y_col], **kwargs)
        
        # Add metadata as labels
        title = kwargs.get('title', f'{x_signal.metadata.name} vs {y_signal.metadata.name}')
        self.set_title(figure, title)
        
        x_label = kwargs.get('x_label')
        if not x_label and x_signal.metadata.units:
            try:
                x_label = f"{x_signal.metadata.name or 'X'} ({x_signal.metadata.units.name})"
            except AttributeError:
                x_label = f"{x_signal.metadata.name or 'X'} ({x_signal.metadata.units})"
        
        y_label = kwargs.get('y_label')
        if not y_label and y_signal.metadata.units:
            try:
                y_label = f"{y_signal.metadata.name or 'Y'} ({y_signal.metadata.units.name})"
            except AttributeError:
                y_label = f"{y_signal.metadata.name or 'Y'} ({y_signal.metadata.units})"
        
        self.set_axis_labels(figure, x_label, y_label)
        
        return figure
    
    def visualize_signal(self, signal: SignalData, **kwargs) -> Any:
        """
        Visualize a single signal with automatic plot type selection.
        
        Args:
            signal: The signal to visualize
            **kwargs: Optional visualization parameters
            
        Returns:
            A backend-specific figure object
        """
        # For now, default to time series plot
        return self.create_time_series_plot(signal, **kwargs)
    
    def visualize_collection(self, collection: SignalCollection, 
                           signals: Optional[List[str]] = None,
                           layout: Optional[str] = "vertical", 
                           **kwargs) -> Any:
        """
        Visualize multiple signals from a collection.
        
        Args:
            collection: The signal collection
            signals: Optional list of signal keys or base names to visualize (if None, visualize all)
            layout: Layout type, one of "vertical", "horizontal", "grid"
            **kwargs: Optional visualization parameters
                - shared_x_range: If True, synchronize x-axes across all plots
                - time_range: Optional tuple of (start_time, end_time) to restrict the view
                - subplot_titles: List of titles for subplots (optional)
                - strict: If False, skip missing signals with a warning instead of raising an error
            
        Returns:
            A backend-specific figure or layout object
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle signal selection
        if signals is None:
            # Get all signals from the collection
            signals = list(collection.signals.keys())
            logger.info(f"Using all {len(signals)} signals from collection")
        else:
            # Process signal specifiers which could be keys or base names
            expanded_signals = []
            for signal_spec in signals:
                # Try to get signals by base name or specific key
                matching_signals = collection.get_signals(input_spec=signal_spec)
                
                if matching_signals:
                    # If we got signals by base name, we'll have SignalData objects
                    # We need to find their keys in the collection
                    for matching_signal in matching_signals:
                        # Find the key for this signal in the collection
                        for key, signal in collection.signals.items():
                            if signal is matching_signal:
                                expanded_signals.append(key)
                                break
                else:
                    # If it's an exact key in the collection, use it directly
                    if signal_spec in collection.signals:
                        expanded_signals.append(signal_spec)
                    else:
                        # No matches found for this specifier
                        strict = kwargs.get('strict', True)
                        if strict:
                            raise ValueError(f"No signals found for specifier '{signal_spec}'")
                        else:
                            logger.warning(f"No signals found for specifier '{signal_spec}', skipping")
            
            signals = expanded_signals
            logger.info(f"Found {len(signals)} signals matching the provided specifiers")
        
        if not signals:
            logger.warning("No signals selected for visualization")
            # Create an empty figure with a message
            fig = self.create_figure(**kwargs)
            # Add a text annotation saying "No signals to display"
            return fig
            
        logger.info(f"Visualizing collection with {len(signals)} signals using {layout} layout")
        
        # Generate subplot titles if not explicitly provided
        subplot_titles = kwargs.get('subplot_titles', [])
        if not subplot_titles and len(signals) > 1:
            logger.debug("Generating subplot titles from signal metadata")
            subplot_titles = []
            for signal_key in signals:
                try:
                    signal = collection.get_signal(signal_key)
                    
                    # Start with the signal name
                    title_parts = []
                    
                    # Add sensor model if available
                    if hasattr(signal.metadata, 'sensor_model') and signal.metadata.sensor_model:
                        model = signal.metadata.sensor_model
                        model_name = model.name if hasattr(model, 'name') else str(model)
                        title_parts.append(model_name)
                    
                    # Add signal name
                    if signal.metadata.name:
                        title_parts.append(signal.metadata.name)
                    else:
                        title_parts.append(signal_key)
                    
                    # Add body position if available
                    if hasattr(signal.metadata, 'body_position') and signal.metadata.body_position:
                        position = signal.metadata.body_position
                        position_name = position.name if hasattr(position, 'name') else str(position)
                        title_parts.append(f"({position_name})")
                    
                    title = " ".join(title_parts)
                    subplot_titles.append(title)
                except Exception as e:
                    logger.warning(f"Could not generate title for signal '{signal_key}': {str(e)}")
                    subplot_titles.append(signal_key)
            
            # Add subplot_titles to kwargs for use by visualizer implementations
            kwargs['subplot_titles'] = subplot_titles
        
        # Determine common time range if needed
        link_x_axes = kwargs.get('link_x_axes', True)
        
        if link_x_axes and len(signals) > 1:
            # Find common time bounds across all signals
            min_times = []
            max_times = []
            
            for signal_key in signals:
                try:
                    signal = collection.get_signal(signal_key)
                    data = signal.get_data()
                    if data is not None and len(data) > 0:
                        min_times.append(data.index.min())
                        max_times.append(data.index.max())
                except Exception as e:
                    logger.warning(f"Could not get time range for signal '{signal_key}': {str(e)}")
            
            if min_times and max_times:
                # Use the intersection of all time ranges
                common_start = max(min_times)
                common_end = min(max_times)
                
                logger.info(f"Using common time range: {common_start} to {common_end}")
                
                # Override time range in kwargs if specified
                if 'time_range' in kwargs:
                    user_start, user_end = kwargs['time_range']
                    common_start = user_start if user_start else common_start
                    common_end = user_end if user_end else common_end
                
                kwargs['x_min'] = common_start
                kwargs['x_max'] = common_end
        
        # Create figures for each signal
        figures = []
        for signal_key in signals:
            try:
                signal = collection.get_signal(signal_key)
                # Add signal key to the title if not already specified
                if 'title' not in kwargs:
                    title = signal.metadata.name or signal_key
                    signal_kwargs = {**kwargs, 'title': title}
                else:
                    signal_kwargs = kwargs

                # Check if this signal is categorical and might need a default y-range
                is_categorical_signal = False
                primary_col = signal.get_data().columns[0]
                signal_data = signal.get_data()
                if pd.api.types.is_categorical_dtype(signal_data[primary_col]) or \
                   pd.api.types.is_object_dtype(signal_data[primary_col]) or \
                   (hasattr(signal, 'signal_type') and signal.signal_type == SignalType.EEG_SLEEP_STAGE):
                    is_categorical_signal = True

                # If categorical and no y-limits are set, provide a default for Bokeh
                # This helps when it's the only plot or determines linked axis range
                if is_categorical_signal and 'y_min' not in signal_kwargs and 'y_max' not in signal_kwargs:
                     # Set a default range like (-0.5, 0.5) or (0, 1)
                     # Let's use (0, 1) as it matches the default in add_categorical_regions
                     signal_kwargs.setdefault('y_min', 0)
                     signal_kwargs.setdefault('y_max', 1)
                     logger.debug(f"Setting default y-range (0, 1) for categorical signal '{signal_key}'")

                figure = self.visualize_signal(signal, **signal_kwargs)
                figures.append(figure)
            except Exception as e:
                import warnings
                warnings.warn(f"Could not visualize signal '{signal_key}': {str(e)}")
                logger.error(f"Could not visualize signal '{signal_key}': {str(e)}")
        
        # Apply the layout
        if not figures:
            raise ValueError("No figures created - check signal keys")
            
        if layout == "vertical":
            return self.create_grid_layout(figures, len(figures), 1, **kwargs)
        elif layout == "horizontal":
            return self.create_grid_layout(figures, 1, len(figures), **kwargs)
        elif layout == "grid":
            # Calculate a reasonable grid size based on the number of figures
            import math
            cols = math.ceil(math.sqrt(len(figures)))
            rows = math.ceil(len(figures) / cols)
            return self.create_grid_layout(figures, rows, cols, **kwargs)
        elif layout == "overlay":
            # For overlaying, return only the first figure but with all signals
            # This requires backend-specific implementation and is not fully supported
            logger.warning("Overlay layout is experimental and may not work with all backends")
            
            # If we're overlaying multiple signals, ensure they have different colors
            if len(figures) > 1 and figures[0]:
                # For each signal after the first, extract its traces and add them to the first figure
                # with different colors
                first_fig = figures[0]
                for i, fig in enumerate(figures[1:], 1):
                    if hasattr(self, 'transfer_traces'):
                        # Use dedicated method if available
                        self.transfer_traces(fig, first_fig, color_index=i)
                    # Color handling for individual signals is done in create_time_series_plot
            return figures[0] if figures else None
        else:
            return figures
    
    def create_from_config(self, config: Dict[str, Any], collection: SignalCollection) -> Any:
        """
        Create a visualization from a configuration dictionary.
        
        Args:
            config: Visualization configuration dictionary
            collection: Signal collection containing the data to visualize
            
        Returns:
            A backend-specific figure or layout object
        """
        plot_type = config.get('type', 'time_series')
        
        # Extract parameters section if it exists
        parameters = config.get('parameters', {})
        
        # Extract only the parameters section from the config without including default params
        # This ensures width/height are only set when explicitly provided in YAML
        viz_params = parameters.copy()
        
        # Add other config options (but not default params)
        for k, v in config.items():
            if k not in ['type', 'signals', 'x_signal', 'y_signal', 
                        'output', 'format', 'layout', 'parameters']:
                viz_params[k] = v
        
        # Default to not strict (skip missing signals with warning)
        if 'strict' not in viz_params:
            viz_params['strict'] = False
        
        if plot_type == 'time_series':
            signal_specifiers = config.get('signals', [])
            layout = config.get('layout', 'vertical')
            return self.visualize_collection(collection, signal_specifiers, layout, **viz_params)
            
        elif plot_type == 'scatter':
            x_key = config.get('x_signal')
            y_key = config.get('y_signal')
            
            if not x_key or not y_key:
                raise ValueError("Scatter plot requires 'x_signal' and 'y_signal'")
            
            # Try to get signals using the flexible input_spec approach
            x_signals = collection.get_signals(input_spec=x_key)
            y_signals = collection.get_signals(input_spec=y_key)
            
            if not x_signals:
                strict = viz_params.get('strict', True)
                if strict:
                    raise ValueError(f"No x-axis signals found for specifier '{x_key}'")
                else:
                    import logging
                    logging.getLogger(__name__).warning(f"No x-axis signals found for specifier '{x_key}', cannot create scatter plot")
                    # Return empty figure with message
                    fig = self.create_figure(**viz_params)
                    return fig
            
            if not y_signals:
                strict = viz_params.get('strict', True)
                if strict:
                    raise ValueError(f"No y-axis signals found for specifier '{y_key}'")
                else:
                    import logging
                    logging.getLogger(__name__).warning(f"No y-axis signals found for specifier '{y_key}', cannot create scatter plot")
                    # Return empty figure with message
                    fig = self.create_figure(**viz_params)
                    return fig
                
            # Use the first signal from each group
            x_signal = x_signals[0]
            y_signal = y_signals[0]
            
            return self.create_scatter_plot(x_signal, y_signal, **viz_params)
            
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def process_visualization_config(self, config: Dict[str, Any], 
                                    collection: SignalCollection) -> None:
        """
        Process a visualization configuration and save the result.
        
        Args:
            config: Visualization configuration dictionary
            collection: Signal collection containing the data to visualize
        """
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Processing visualization config: {config.get('title', 'Untitled')}")
    
        # Extract parameters from config
        params = config.get('parameters', {})
    
        # Log key parameters for debugging
        if 'width' in params or 'height' in params:
            logger.debug(f"Visualization dimensions in config: width={params.get('width')}, height={params.get('height')}")
                
            # Ensure these dimensions are propagated to individual plots
            if 'signals' in config and config.get('layout') in ['vertical', 'horizontal', 'grid']:
                logger.debug(f"Ensuring dimensions are applied to multi-plot layout")
        
        figure = self.create_from_config(config, collection)
        
        # Save output if specified
        if 'output' in config:
            # Ensure the directory exists
            output_path = config['output']
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            format = config.get('format')
            if not format and '.' in output_path:
                # Try to infer format from file extension
                format = output_path.split('.')[-1].lower()
            
            format = format or 'html'  # Default to HTML
            logger.info(f"Saving visualization to {output_path} in {format} format")
            self.save(figure, output_path, format)
        else:
            # If no output file specified, just show the figure
            logger.info("No output file specified, displaying figure")
            self.show(figure)
