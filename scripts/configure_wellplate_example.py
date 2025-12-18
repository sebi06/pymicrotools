from ome_zarr_utils import PlateConfiguration, PlateType, PLATE_FORMATS
from well_surface import Point3D

# Create a 96-well plate configuration
plate = PlateConfiguration(8, 12, "well96")
print(f"Plate Name: {plate.name}")
print(f"Rows: {plate.rows}, Columns: {plate.columns}, Total Wells: {plate.total_wells}")
print(f"Row Labels: {plate.row_labels}")
print(f"Column Labels: {plate.column_labels}")

print("\n" + "=" * 70)
print("Adding Surface Calibration Points")
print("=" * 70)

# Define calibration points that define the surface
# These represent focus/height measurements at different positions on the plate
P1 = Point3D(X=0.0, Y=0.0, Z=0.0)
P2 = Point3D(X=0.0, Y=96000.0, Z=80.0)
P3 = Point3D(X=108000.0, Y=96000.0, Z=150.0)
P4 = Point3D(X=108000.0, Y=0.0, Z=200.0)
P5 = Point3D(X=54000.0, Y=48000.0, Z=100.0)

calibration_points = [P1, P2, P3, P4, P5]

# Add all calibration points to the plate
for pt in calibration_points:
    plate.add_surface_point(pt.X, pt.Y, pt.Z)

# Print surface calibration information
plate.print_surface_info()

print("\n" + "=" * 70)
print("Sample Z-Value Queries")
print("=" * 70)

# Query Z-values at various positions
query_points = [
    (0.0, 0.0, "Top-left corner"),
    (54000.0, 48000.0, "Center of plate"),
    (108000.0, 96000.0, "Bottom-right corner"),
    (120000.0, 100000.0, "Outside plate (extrapolated)"),
]

for x, y, description in query_points:
    z = plate.get_z_for_xy(x, y)
    print(f"  XY ({x:10.1f}, {y:10.1f}) → Z = {z:10.2f}  [{description}]")

print("\n" + "=" * 70)
print("3D Surface Visualization")
print("=" * 70)

# Visualize the 3D surface using the surface interpolator
if plate.has_surface_calibration():
    # Generate test points to visualize
    test_visualization_points = [
        (0.0, 0.0),
        (27000.0, 24000.0),
        (54000.0, 48000.0),
        (81000.0, 72000.0),
        (108000.0, 96000.0),
        (120000.0, 100000.0),  # Outside hull for extrapolation demo
        (30000.0, 80000.0),  # Extra points for visualization
        (90000.0, 40000.0),
    ]

    # Create the visualization
    print("\nGenerating 3D surface plot...")
    print("  - Red dots: Calibration points (control points)")
    print("  - Blue triangles: Points inside the convex hull (interpolated)")
    print(
        "  - Orange inverted triangles: Points outside the convex hull (extrapolated)"
    )
    print("  - Viridis colored surface: Interpolated surface inside hull")
    print("  - Gray surface: Extrapolated surface outside hull")

    plate._surface_interpolator.visualize_surface(
        interpolation_density=25,
        test_points=test_visualization_points,
        title=f"3D Surface Interpolation - {plate.name} (96-Well Plate)",
        figsize=(14, 10),
        show_extrapolation=True,
    )
else:
    print("ERROR: Surface calibration not available. Cannot visualize.")
