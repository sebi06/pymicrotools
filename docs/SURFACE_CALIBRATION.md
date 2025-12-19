# PlateConfiguration with Surface Calibration

## Overview

The `PlateConfiguration` class has been extended to support surface calibration using arbitrary XYZ control points. This allows you to interpolate Z-values (focus heights) for any XY position within your microplate.

## What's New

### New Features

1. **Surface Point Storage**: Store calibration points (X, Y, Z coordinates) in the plate configuration
2. **Automatic Interpolation**: Use Delaunay triangulation and cubic spline interpolation for smooth Z-value calculation
3. **Extrapolation Support**: Extend beyond the calibration boundary using a best-fit plane
4. **Easy API**: Simple methods to add points and query Z-values

## How to Use

### Basic Usage

```python
from ome_zarr_utils import PlateType

# Get a 96-well plate configuration
plate = PlateType.PLATE_96.value

# Add calibration points (X, Y, Z in micrometers)
plate.add_surface_points([
    (0.0, 0.0, 0.0),           # Top-left
    (0.0, 96000.0, 80.0),      # Top-right
    (108000.0, 96000.0, 150.0),  # Bottom-right
])

# Query Z-value for any XY position
z = plate.get_z_for_xy(54000.0, 48000.0)  # Center of plate → Z = 75.0

# Display calibration info
plate.print_surface_info()
```

### Adding Points Incrementally

```python
plate = PlateType.PLATE_24.value

# Add points one at a time
plate.add_surface_point(0.0, 0.0, 0.0)
plate.add_surface_point(0.0, 27000.0, 50.0)
plate.add_surface_point(54000.0, 27000.0, 100.0)

# Surface interpolator is automatically initialized after 3+ points
if plate.has_surface_calibration():
    z = plate.get_z_for_xy(27000.0, 13500.0)
```

### Custom Plate with Built-in Calibration

```python
from ome_zarr_utils import PlateConfiguration

plate = PlateConfiguration(
    rows=10,
    columns=10,
    name="Custom 100-Well Plate",
    surface_points=[
        (0.0, 0.0, 0.0),
        (0.0, 90000.0, 100.0),
        (90000.0, 90000.0, 150.0),
    ]
)

# Interpolator is ready to use immediately
z = plate.get_z_for_xy(45000.0, 45000.0)
```

## API Reference

### Methods

#### `add_surface_point(x: float, y: float, z: float) -> None`
Add a single calibration point. If 3+ points are present, the surface interpolator is automatically initialized.

```python
plate.add_surface_point(0.0, 0.0, 0.0)
```

#### `add_surface_points(points: List[tuple]) -> None`
Add multiple calibration points at once.

```python
plate.add_surface_points([
    (0.0, 0.0, 0.0),
    (0.0, 96000.0, 80.0),
    (108000.0, 96000.0, 150.0),
])
```

#### `has_surface_calibration() -> bool`
Check if surface calibration is available.

```python
if plate.has_surface_calibration():
    z = plate.get_z_for_xy(x, y)
```

#### `get_z_for_xy(x: float, y: float) -> Optional[float]`
Get Z-coordinate for any XY position.

- **Inside convex hull**: Uses cubic spline interpolation for smooth results
- **Outside convex hull**: Uses best-fit plane extrapolation
- **No calibration**: Returns `None` and logs a warning

```python
z = plate.get_z_for_xy(54000.0, 48000.0)  # Returns interpolated Z
z = plate.get_z_for_xy(120000.0, 100000.0)  # Returns extrapolated Z
```

#### `get_z_for_well(well_row: int, well_column: int) -> Optional[float]`
Get Z-coordinate for a specific well position (0-based indices).

```python
z = plate.get_z_for_well(3, 5)  # Get Z for well E6
```

#### `print_surface_info() -> None`
Display detailed information about surface calibration.

```python
plate.print_surface_info()
```

Output:
```
======================================================================
Surface Calibration for 96-Well Plate
======================================================================
Number of calibration points: 3

Calibration Points (X, Y, Z):
  P1: X=        0.00, Y=        0.00, Z=        0.00
  P2: X=        0.00, Y=    96000.00, Z=       80.00
  P3: X=   108000.00, Y=    96000.00, Z=      150.00

Plane Equation: Z = 0.00064815*X + 0.00083333*Y + -0.00000000
======================================================================
```

## Properties

### `surface_points: List[tuple]`
Access the list of calibration points.

```python
print(plate.surface_points)  # [(0.0, 0.0, 0.0), (0.0, 96000.0, 80.0), ...]
```

### `total_wells: int`
Total number of wells in the plate.

```python
print(plate.total_wells)  # 96 for 96-well plate
```

### `row_labels: list`
Row labels (A, B, C, ...).

```python
print(plate.row_labels)  # ['A', 'B', ..., 'H'] for 96-well
```

### `column_labels: list`
Column labels (1, 2, 3, ...).

```python
print(plate.column_labels)  # ['1', '2', ..., '12'] for 96-well
```

## Technical Details

### Surface Interpolation Algorithm

**Inside Convex Hull:**
- Uses **Clough-Tocher cubic spline interpolation**
- Provides smooth, continuous surface with C¹ continuity
- Based on Delaunay triangulation of the XY plane

**Outside Convex Hull:**
- Uses **best-fit plane equation**: $Z = a \cdot X + b \cdot Y + c$
- Fitted using least squares method
- Ensures natural surface extension beyond calibration boundary

### Surface Points Integration

The extended `PlateConfiguration` class:
1. Accepts arbitrary 3D points via `add_surface_point()` or `add_surface_points()`
2. Automatically initializes the `well_surface.Surface3D` interpolator when 3+ points are available
3. Provides convenient query methods (`get_z_for_xy()`, `get_z_for_well()`)
4. Maintains backward compatibility with existing code

## Example Workflow

```python
from ome_zarr_utils import PlateType

# 1. Create plate configuration
plate = PlateType.PLATE_96.value

# 2. Add calibration data from your measurements
# (e.g., focus/Z-values at known positions)
calibration_data = [
    (0.0, 0.0, 50.0),        # Measured at top-left
    (54000.0, 48000.0, 75.0), # Measured at center
    (108000.0, 96000.0, 100.0),  # Measured at bottom-right
]
plate.add_surface_points(calibration_data)

# 3. Display calibration info
plate.print_surface_info()

# 4. For each well in your experiment, get the calibrated Z-value
for row in range(plate.rows):
    for col in range(plate.columns):
        z = plate.get_z_for_well(row, col)
        # Use this Z-value in your microscopy acquisition
        print(f"Well ({row}, {col}): Focus Z = {z:.2f}")
```

## Files Modified/Created

- **Modified**: `/scripts/ome_zarr_utils.py` - Extended `PlateConfiguration` class
- **Created**: `/scripts/surface_calibration_example.py` - Example usage demonstrations

## Notes

- Minimum 3 control points required for surface interpolation
- Coordinates should be in micrometers (μm)
- Z-values can be any measurement unit (focus distance, height, etc.)
- The surface automatically adapts when points are added
- All Z-values are valid (no NaN) - extrapolation provides smooth continuation
