"""
Example: Using PlateConfiguration with Surface Calibration

This example demonstrates how to add XYZ calibration points to a plate configuration
and use them to interpolate Z-values for any XY position.
"""

from ome_zarr_utils import PlateConfiguration, PlateType
from well_surface import Point3D


def example_basic_usage():
    """Basic example: Add surface points and query Z-values"""
    print("=" * 70)
    print("Example 1: Basic Surface Calibration")
    print("=" * 70)

    # Create a 96-well plate
    plate = PlateType.PLATE_96.value

    # Add calibration points (X, Y, Z coordinates in micrometers)
    # These represent focus heights at different positions on the plate
    calibration_points = [
        (0.0, 0.0, 0.0),  # Top-left corner
        (0.0, 96000.0, 80.0),  # Top-right corner
        (108000.0, 96000.0, 150.0),  # Bottom-right corner
    ]

    plate.add_surface_points(calibration_points)

    # Display surface information
    plate.print_surface_info()

    # Query Z-values at various positions
    print("\nInterpolated Z-Values:")
    print("-" * 70)
    query_points = [
        (0.0, 0.0, "Top-left corner"),
        (54000.0, 48000.0, "Center of plate"),
        (108000.0, 96000.0, "Bottom-right corner"),
        (120000.0, 100000.0, "Outside plate (extrapolated)"),
    ]

    for x, y, description in query_points:
        z = plate.get_z_for_xy(x, y)
        print(f"  XY ({x:10.1f}, {y:10.1f}) → Z = {z:10.2f}  [{description}]")


def example_well_specific():
    """Example: Get Z-value for specific well positions"""
    print("\n" + "=" * 70)
    print("Example 2: Z-Values for Specific Wells")
    print("=" * 70)

    # Create a 96-well plate
    plate = PlateType.PLATE_96.value

    # Add calibration points
    plate.add_surface_points(
        [
            (0.0, 0.0, 0.0),
            (0.0, 96000.0, 80.0),
            (108000.0, 96000.0, 150.0),
        ]
    )

    print(f"Plate: {plate.name}")
    print(f"Dimensions: {plate.rows} rows × {plate.columns} columns")

    # Get Z-values for specific wells
    print("\nZ-Values for Selected Wells:")
    print("-" * 70)
    wells = [
        (0, 0, "A1 (top-left)"),
        (0, 11, "A12 (top-right)"),
        (7, 0, "H1 (bottom-left)"),
        (7, 11, "H12 (bottom-right)"),
        (3, 5, "E6 (center)"),
    ]

    for row, col, description in wells:
        z = plate.get_z_for_well(row, col)
        print(f"  Well {description}: Z = {z:10.2f}")


def example_incremental_addition():
    """Example: Adding surface points incrementally"""
    print("\n" + "=" * 70)
    print("Example 3: Incremental Surface Point Addition")
    print("=" * 70)

    # Create a 24-well plate
    plate = PlateType.PLATE_24.value

    print("Adding calibration points incrementally:")
    print("-" * 70)

    # Add points one by one
    plate.add_surface_point(0.0, 0.0, 0.0)
    print(
        f"After P1: {len(plate.surface_points)} point(s) - Surface calibration: {plate.has_surface_calibration()}"
    )

    plate.add_surface_point(0.0, 27000.0, 50.0)
    print(
        f"After P2: {len(plate.surface_points)} point(s) - Surface calibration: {plate.has_surface_calibration()}"
    )

    plate.add_surface_point(54000.0, 27000.0, 100.0)
    print(
        f"After P3: {len(plate.surface_points)} point(s) - Surface calibration: {plate.has_surface_calibration()}"
    )

    # Now we have surface calibration
    if plate.has_surface_calibration():
        plate.print_surface_info()

        # Query some points
        z = plate.get_z_for_xy(27000.0, 13500.0)
        print(f"\nInterpolated Z at center: {z:.2f}")


def example_custom_plate_with_calibration():
    """Example: Create a plate configuration with immediate calibration"""
    print("\n" + "=" * 70)
    print("Example 4: 384-Well Plate with Built-in Calibration")
    print("=" * 70)

    # Use a standard 384-well plate (16 rows × 24 columns)
    plate = PlateType.PLATE_384.value

    # Add surface calibration points
    plate.add_surface_points(
        [
            (0.0, 0.0, 0.0),
            (0.0, 216000.0, 100.0),  # 23 columns × 9000 μm
            (135000.0, 216000.0, 150.0),  # 15 rows × 9000 μm
        ]
    )

    plate.print_surface_info()

    # Query points
    print("\nSample Queries:")
    print("-" * 70)
    z_center = plate.get_z_for_xy(67500.0, 108000.0)
    print(f"Center of plate: Z = {z_center:.2f}")

    z_corner = plate.get_z_for_xy(0.0, 0.0)
    print(f"Corner of plate: Z = {z_corner:.2f}")

    # Query using well IDs
    print("\nUsing Well IDs:")
    print("-" * 70)
    z_a1 = plate.get_z_for_well_id("A1")
    print(f"Well A1: Z = {z_a1:.2f}")

    z_center_well = plate.get_z_for_well_id("H12")
    print(f"Well H12: Z = {z_center_well:.2f}")

    z_edge = plate.get_z_for_well_id("P24")
    print(f"Well P24 (bottom-right): Z = {z_edge:.2f}")


def example_point3d_objects():
    """Example: Using Point3D objects directly for surface calibration"""
    print("\n" + "=" * 70)
    print("Example 5: Surface Calibration with Point3D Objects")
    print("=" * 70)

    # Create a 48-well plate
    plate = PlateType.PLATE_48.value

    # Create calibration points using Point3D objects
    calibration_points = [
        Point3D(X=0.0, Y=0.0, Z=0.0),  # Top-left
        Point3D(X=0.0, Y=54000.0, Z=75.0),  # Top-right
        Point3D(X=60000.0, Y=54000.0, Z=150.0),  # Bottom-right
    ]

    plate.add_surface_points(calibration_points)

    plate.print_surface_info()

    # Query Z-values
    print("\nInterpolated Z-Values:")
    print("-" * 70)
    test_points = [
        (0.0, 0.0, "Top-left"),
        (30000.0, 27000.0, "Center"),
        (60000.0, 54000.0, "Bottom-right"),
    ]

    for x, y, description in test_points:
        z = plate.get_z_for_xy(x, y)
        print(f"  XY ({x:10.1f}, {y:10.1f}) → Z = {z:10.2f}  [{description}]")


def example_well_id_queries():
    """Example: Query Z-values using well IDs"""
    print("\n" + "=" * 70)
    print("Example 6: Query Z-Values Using Well IDs")
    print("=" * 70)

    # Create a 96-well plate with calibration
    plate = PlateType.PLATE_96.value
    plate.add_surface_points(
        [
            (0.0, 0.0, 0.0),
            (0.0, 96000.0, 80.0),
            (108000.0, 96000.0, 150.0),
        ]
    )

    print(f"Plate: {plate.name}")
    print(f"Dimensions: {plate.rows} rows × {plate.columns} columns\n")

    # Query using different well ID formats
    well_ids = [
        ("A1", "Well A1 (top-left)"),
        ("A/1", "Well A/1 (alternative format)"),
        ("A-1", "Well A-1 (dash format)"),
        ("A12", "Well A12 (top-right)"),
        ("H1", "Well H1 (bottom-left)"),
        ("H12", "Well H12 (bottom-right)"),
        ("E6", "Well E6 (center)"),
    ]

    print("Z-Values for Wells:")
    print("-" * 70)
    for well_id, description in well_ids:
        try:
            z = plate.get_z_for_well_id(well_id)
            print(f"  {well_id:6s} → Z = {z:10.2f}  [{description}]")
        except ValueError as e:
            print(f"  {well_id:6s} → ERROR: {e}")

    # Test error handling
    print("\nError Handling Examples:")
    print("-" * 70)
    invalid_ids = [
        ("I1", "Invalid row (beyond H)"),
        ("A13", "Invalid column (beyond 12)"),
        ("AB1", "Invalid format (two letters)"),
        ("Z99", "Invalid row and column"),
    ]

    for well_id, description in invalid_ids:
        try:
            z = plate.get_z_for_well_id(well_id)
            print(f"  {well_id:6s} → Z = {z:.2f}")
        except ValueError as e:
            print(f"  {well_id:6s} → ✗ {description}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_well_specific()
    example_incremental_addition()
    example_custom_plate_with_calibration()
    example_point3d_objects()
    example_well_id_queries()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
