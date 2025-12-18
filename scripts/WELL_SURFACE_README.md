# 3D Surface Interpolation and Extrapolation

## Overview

The `well_surface.py` module provides tools to define and analyze 3D surfaces from control points. It supports smooth interpolation within the surface boundaries and intelligent extrapolation beyond them using a best-fit plane equation.

## Key Features

### 1. **Point3D Dataclass**

A simple dataclass to represent 3D coordinates with X, Y, and Z values.

```python
P1 = Point3D(X=0.0, Y=0.0, Z=0.0)
P2 = Point3D(X=0.0, Y=8000.0, Z=80.0)
P3 = Point3D(X=12000.0, Y=8000.0, Z=150.0)
```

### 2. **Surface3D Class**

The main class for surface management with interpolation and extrapolation capabilities.

#### Initialization

```python
control_points = [P1, P2, P3]
surface = Surface3D(control_points)
```

**Requirements:** Minimum 3 control points needed to define a valid surface.

## Interpolation Methods

### Cubic Spline Interpolation (Inside Convex Hull)

For points within the convex hull of control points, the surface uses **Clough-Tocher cubic spline interpolation**, which provides:

- **Smooth, continuous surfaces** with C¹ continuity (continuous first derivatives)
- **No visible faceting** between triangular elements
- **Accurate Z-values** at all control points

```python
z = surface.get_z_at_xy(6000.0, 4000.0)  # Returns Z = 75.00
```

### Plane-Based Extrapolation (Outside Convex Hull)

For points beyond the convex hull, the surface uses a **best-fit plane equation**:

$$Z = a \cdot X + b \cdot Y + c$$

This plane is calculated using least squares fitting and ensures:

- **Natural surface extension** following the overall trend
- **Continuity** between interpolated and extrapolated regions
- **Predictable Z-values** based on the surface slope

```python
z = surface._extrapolate_z(15000.0, 8000.0)  # Returns extrapolated Z value
```

## Example Usage

### Basic Workflow

```python
from well_surface import Point3D, Surface3D

# 1. Define control points
P1 = Point3D(X=0.0, Y=0.0, Z=0.0)
P2 = Point3D(X=0.0, Y=8000.0, Z=80.0)
P3 = Point3D(X=12000.0, Y=8000.0, Z=150.0)

# 2. Create surface
control_points = [P1, P2, P3]
surface = Surface3D(control_points)

# 3. Query Z-values for any XY position
# Inside convex hull - uses interpolation
z_inside = surface.get_z_at_xy(6000.0, 4000.0)  # Z = 75.00 [INSIDE]

# Outside convex hull - uses extrapolation
try:
    z = surface.get_z_at_xy(15000.0, 8000.0)
except ValueError:
    z = surface._extrapolate_z(15000.0, 8000.0)  # Z = 167.50 [OUTSIDE]

# 4. Visualize the surface
surface.visualize_surface(
    interpolation_density=30,
    test_points=[(6000.0, 4000.0), (15000.0, 8000.0)],
    show_extrapolation=True
)
```

## Surface Information

Print detailed information about the surface:

```python
surface.print_surface_info()
```

Output:

```txt
============================================================
3D Surface Information
============================================================
Number of control points: 3

Control Points:
  P1: Point3D(X=0.0, Y=0.0, Z=0.0)
  P2: Point3D(X=0.0, Y=8000.0, Z=80.0)
  P3: Point3D(X=12000.0, Y=8000.0, Z=150.0)
============================================================
```

## Visualization

### visualize_surface() Method

Creates an interactive 3D plot showing:

**Parameters:**

- `interpolation_density` (int): Resolution of mesh grid (default: 20)
- `test_points` (List[Tuple]): XY coordinates to display as markers
- `title` (str): Plot title
- `figsize` (Tuple): Figure dimensions (width, height)
- `show_extrapolation` (bool): Show extrapolated regions (default: True)

**Visual Elements:**

1. **Interpolated Surface** (Viridis colors)
   - Covers the region inside the convex hull
   - Smooth cubic spline surface
2. **Extrapolated Surface** (Gray, semi-transparent)
   - Extends beyond the convex hull
   - Based on the best-fit plane equation
   - Shows how the surface naturally extends
3. **Control Points** (Red circles with labels)
   - Original 3D points used to define the surface
   - Exact Z-values match at these locations
4. **Test Points**
   - **Blue up-triangles**: Points inside convex hull (interpolated)
   - **Orange down-triangles**: Points outside convex hull (extrapolated)

### Example Visualization Call

```python
surface.visualize_surface(
    interpolation_density=30,
    test_points=[
        (3000.0, 3000.0),    # Inside
        (6000.0, 5000.0),    # Inside
        (12500.0, 4000.0),   # Outside
        (3500.0, -200.0),    # Outside
    ],
    title="3D Surface Interpolation - Well Surface Example",
    show_extrapolation=True
)
```

## Methods Reference

### Public Methods

| Method | Description | Parameters | Returns |
| --- | --- | --- | --- |
| `get_z_at_xy(x, y)` | Get Z-value for XY position inside hull | `x`, `y` floats | Z float value |
| `get_point_on_surface(x, y)` | Get 3D point for XY position inside hull | `x`, `y` floats | Point3D object |
| `print_surface_info()` | Print surface details | — | Prints to console |
| `visualize_surface(...)` | Create interactive 3D visualization | See parameters above | Displays plot |

### Private Methods

| Method | Description |
| --- | --- |
| `_setup_interpolator()` | Initialize interpolation/extrapolation models |
| `_calculate_plane_equation()` | Fit best-fit plane to control points |
| `_extrapolate_z(x, y)` | Calculate Z using plane equation |

## Example: Complete Workflow

```python
#!/usr/bin/env python
"""Example: Complete 3D surface analysis workflow"""

from well_surface import Point3D, Surface3D

# Define control points
P1 = Point3D(X=0.0, Y=0.0, Z=0.0)
P2 = Point3D(X=0.0, Y=8000.0, Z=80.0)
P3 = Point3D(X=12000.0, Y=8000.0, Z=150.0)

# Create surface
surface = Surface3D([P1, P2, P3])

# Display information
surface.print_surface_info()

# Query multiple points
test_points = [
    (3000.0, 3000.0),    # Inside
    (6000.0, 5000.0),    # Inside
    (12500.0, 4000.0),   # Outside
    (3500.0, -200.0),    # Outside
]

print("\nInterpolation/Extrapolation Results:")
print("-" * 60)
for x, y in test_points:
    try:
        z = surface.get_z_at_xy(x, y)
        status = "INSIDE"
    except ValueError:
        z = surface._extrapolate_z(x, y)
        status = "OUTSIDE"
    print(f"XY ({x:8.1f}, {y:8.1f}) → Z = {z:8.2f} [{status}]")

# Visualize
surface.visualize_surface(
    interpolation_density=30,
    test_points=test_points,
    show_extrapolation=True
)
```

## Technical Details

### Interpolation Algorithm: Clough-Tocher

- **Method**: Piecewise cubic polynomial interpolation
- **Triangulation**: Delaunay triangulation on XY plane
- **Continuity**: C¹ (smooth first derivatives)
- **Advantages**:
  - Smooth transitions between triangles
  - No visible faceting
  - More natural-looking surfaces

### Extrapolation Algorithm: Best-Fit Plane

- **Equation**: $Z = a \cdot X + b \cdot Y + c$
- **Fitting**: Least squares method on all control points
- **Properties**:
  - Passes through or near all control points
  - Represents the overall surface trend
  - Extends naturally beyond convex hull

## Use Cases

1. **Well Surface Analysis**: Model microscope well surfaces in high-content screening
2. **Topographic Mapping**: Interpolate elevation data across regions
3. **Device Surfaces**: Model optical or mechanical surface properties
4. **Scientific Data**: Analyze gradients in 3D experimental data

## Requirements

- NumPy (matrix operations)
- SciPy (Delaunay triangulation, interpolation)
- Matplotlib (visualization)

## Notes

- All Z-values are calculated automatically—no NaN values for outside points
- The plane equation ensures physically meaningful extrapolation
- Visualization clearly distinguishes interpolated vs extrapolated regions
- Surface extends infinitely in the extrapolated regions based on the plane equation
