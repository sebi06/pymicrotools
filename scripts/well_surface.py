"""
3D Surface Interpolation using Shapely

This module provides tools to define a 3D surface from control points
and interpolate Z-coordinates for arbitrary XY positions on that surface.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Point3D:
    """Dataclass representing a 3D point with XYZ coordinates."""

    X: float
    Y: float
    Z: float

    def __repr__(self) -> str:
        """Return a readable string representation of the point."""
        return f"Point3D(X={self.X}, Y={self.Y}, Z={self.Z})"

    def to_array(self) -> np.ndarray:
        """Convert point to numpy array format."""
        return np.array([self.X, self.Y, self.Z])


class Surface3D:
    """
    Represents a 3D surface defined by control points.

    Uses Delaunay triangulation and cubic spline interpolation (Clough-Tocher)
    to estimate Z-coordinates for arbitrary XY positions on the surface.
    This provides smooth, continuous surface interpolation.
    """

    def __init__(self, control_points: List[Point3D]):
        """
        Initialize the surface with control points.

        Args:
            control_points (List[Point3D]): List of 3D points defining the surface.
                                           Minimum 3 points required for a valid surface.

        Raises:
            ValueError: If fewer than 3 control points are provided.
        """
        if len(control_points) < 3:
            raise ValueError(
                "At least 3 control points are required to define a surface"
            )

        self.control_points = control_points
        self._setup_interpolator()

    def _setup_interpolator(self) -> None:
        """
        Set up the interpolation model using Delaunay triangulation.

        This creates a triangulated surface from the control points using
        Clough-Tocher (cubic spline) interpolation, allowing efficient and
        smooth Z-value interpolation for arbitrary XY coordinates.
        """
        # Extract XYZ coordinates from control points
        points_array = np.array([p.to_array() for p in self.control_points])

        # Separate XY coordinates for triangulation
        xy_coords = points_array[:, :2]  # Extract X and Y columns
        z_coords = points_array[:, 2]  # Extract Z column

        # Create Delaunay triangulation on XY plane
        self.triangulation = Delaunay(xy_coords)

        # Create cubic spline interpolator (Clough-Tocher) for smooth Z values
        # This provides C1 continuous interpolation (smooth derivatives)
        self.interpolator = CloughTocher2DInterpolator(xy_coords, z_coords)

        # Calculate the best-fit plane for extrapolation beyond convex hull
        self._calculate_plane_equation(xy_coords, z_coords)

    def _calculate_plane_equation(
        self, xy_coords: np.ndarray, z_coords: np.ndarray
    ) -> None:
        """
        Calculate the best-fit plane equation for extrapolation.

        The plane equation is: Z = a*X + b*Y + c
        This plane is fitted to all control points using least squares.
        """
        # Create matrix for least squares fitting: Z = a*X + b*Y + c
        A = np.column_stack([xy_coords[:, 0], xy_coords[:, 1], np.ones(len(xy_coords))])

        # Solve for coefficients using least squares
        coefficients, _, _, _ = np.linalg.lstsq(A, z_coords, rcond=None)

        self.plane_a = coefficients[0]  # Coefficient for X
        self.plane_b = coefficients[1]  # Coefficient for Y
        self.plane_c = coefficients[2]  # Constant term

    def _extrapolate_z(self, x: float, y: float) -> float:
        """
        Extrapolate Z-coordinate using the best-fit plane equation.

        Args:
            x (float): X-coordinate
            y (float): Y-coordinate

        Returns:
            float: Extrapolated Z-coordinate based on plane equation
        """
        return self.plane_a * x + self.plane_b * y + self.plane_c

    def get_z_at_xy(self, x: float, y: float) -> float:
        """
        Interpolate Z-coordinate for a given XY position on the surface.

        Args:
            x (float): X-coordinate of the query point
            y (float): Y-coordinate of the query point

        Returns:
            float: Interpolated Z-coordinate at the given XY position

        Raises:
            ValueError: If the XY point is outside the convex hull of control points
        """
        # Check if the point is within the triangulation bounds
        if not self.triangulation.find_simplex(np.array([x, y])) >= 0:
            raise ValueError(
                f"Point ({x}, {y}) is outside the convex hull of control points"
            )

        # Interpolate Z value at the given XY coordinates
        z_value = self.interpolator(x, y)

        # Handle potential NaN values from interpolation
        if np.isnan(z_value):
            raise ValueError(f"Interpolation failed for point ({x}, {y})")

        return float(z_value)

    def get_point_on_surface(self, x: float, y: float) -> Point3D:
        """
        Get a 3D point on the surface for given XY coordinates.

        Args:
            x (float): X-coordinate of the query point
            y (float): Y-coordinate of the query point

        Returns:
            Point3D: A 3D point with interpolated Z-coordinate
        """
        z = self.get_z_at_xy(x, y)
        return Point3D(X=x, Y=y, Z=z)

    def print_surface_info(self) -> None:
        """Print information about the surface and control points."""
        print("=" * 60)
        print("3D Surface Information")
        print("=" * 60)
        print(f"Number of control points: {len(self.control_points)}")
        print("\nControl Points:")
        for i, point in enumerate(self.control_points, 1):
            print(f"  P{i}: {point}")
        print("=" * 60)

    def visualize_surface(
        self,
        interpolation_density: int = 20,
        test_points: Optional[List[Tuple[float, float]]] = None,
        title: str = "3D Surface Interpolation",
        figsize: Tuple[int, int] = (12, 9),
        show_extrapolation: bool = True,
    ) -> None:
        """
        Visualize the 3D surface with control points and optional interpolated points.

        Args:
            interpolation_density (int): Number of points to generate along each axis
                                        for the surface mesh (default: 20)
            test_points (Optional[List[Tuple[float, float]]]): List of XY coordinates
                                                              to interpolate and display
            title (str): Title for the plot
            figsize (Tuple[int, int]): Figure size (width, height) in inches
            show_extrapolation (bool): Whether to extrapolate beyond convex hull (default: True)
        """
        # Extract control point coordinates
        control_points_array = np.array([p.to_array() for p in self.control_points])
        control_x = control_points_array[:, 0]
        control_y = control_points_array[:, 1]
        control_z = control_points_array[:, 2]

        # Determine the bounds for the mesh grid
        x_min, x_max = control_x.min(), control_x.max()
        y_min, y_max = control_y.min(), control_y.max()

        # Add 10% margin to the bounds for better visualization
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1

        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        # Create a mesh grid for surface visualization
        x_grid = np.linspace(x_min, x_max, interpolation_density)
        y_grid = np.linspace(y_min, y_max, interpolation_density)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate Z values for the mesh grid
        Z = np.zeros_like(X)
        outside_hull_mask = np.zeros_like(X, dtype=bool)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = self.get_z_at_xy(X[i, j], Y[i, j])
                except ValueError:
                    # Point is outside the convex hull
                    outside_hull_mask[i, j] = True
                    if show_extrapolation:
                        # Use the plane equation to extrapolate beyond the convex hull
                        Z[i, j] = self._extrapolate_z(X[i, j], Y[i, j])
                    else:
                        Z[i, j] = np.nan

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot the interpolated surface (inside convex hull)
        Z_inside = Z.copy()
        Z_inside[outside_hull_mask] = np.nan
        surf_inside = ax.plot_surface(
            X,
            Y,
            Z_inside,
            alpha=0.7,
            cmap="viridis",
            edgecolor="none",
            label="Interpolated Surface",
        )

        # Plot extrapolated surface (outside convex hull) with different styling
        if show_extrapolation:
            Z_outside = Z.copy()
            Z_outside[~outside_hull_mask] = np.nan
            ax.plot_surface(
                X,
                Y,
                Z_outside,
                alpha=0.3,
                cmap="Greys",
                edgecolor="none",
                label="Extrapolated Surface",
            )

        # Plot the control points in red
        ax.scatter(
            control_x,
            control_y,
            control_z,
            color="red",
            s=200,
            marker="o",
            edgecolors="darkred",
            linewidth=2,
            label="Control Points",
            zorder=5,
        )

        # Add labels for control points
        for i, point in enumerate(self.control_points, 1):
            ax.text(
                point.X,
                point.Y,
                point.Z,
                f"  P{i}",
                fontsize=10,
                fontweight="bold",
                color="darkred",
            )

        # Plot interpolated test points if provided
        if test_points:
            inside_hull_points = []
            outside_hull_points = []

            for x, y in test_points:
                try:
                    # Try to get Z from interpolation (inside hull)
                    z = self.get_z_at_xy(x, y)
                    inside_hull_points.append([x, y, z])
                except ValueError:
                    # Point is outside convex hull
                    if show_extrapolation:
                        # Use plane-based extrapolation
                        z = self._extrapolate_z(x, y)
                        outside_hull_points.append([x, y, z])

            # Plot points inside convex hull
            if inside_hull_points:
                inside_hull_array = np.array(inside_hull_points)
                ax.scatter(
                    inside_hull_array[:, 0],
                    inside_hull_array[:, 1],
                    inside_hull_array[:, 2],
                    color="blue",
                    s=100,
                    marker="^",
                    edgecolors="darkblue",
                    linewidth=1.5,
                    label="Interpolated Points",
                    zorder=4,
                )

            # Plot points outside convex hull
            if outside_hull_points:
                outside_hull_array = np.array(outside_hull_points)
                ax.scatter(
                    outside_hull_array[:, 0],
                    outside_hull_array[:, 1],
                    outside_hull_array[:, 2],
                    color="orange",
                    s=100,
                    marker="v",
                    edgecolors="darkorange",
                    linewidth=1.5,
                    label="Extrapolated Points",
                    zorder=4,
                )

        # Customize the plot
        ax.set_xlabel("X Coordinate", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y Coordinate", fontsize=11, fontweight="bold")
        ax.set_zlabel("Z Coordinate", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

        # Add colorbar
        cbar = fig.colorbar(surf_inside, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("Z Value", fontsize=10, fontweight="bold")

        # Add legend
        ax.legend(loc="upper left", fontsize=10)

        # Adjust viewing angle for better visualization
        ax.view_init(elev=25, azim=45)

        # Tight layout for better spacing
        plt.tight_layout()

        # Display the plot
        plt.show()


def main():
    """
    Example usage: Create a surface from 3 control points and interpolate Z values.
    """
    # Define 3 control points that define the surface
    P1 = Point3D(X=0.0, Y=0.0, Z=0.0)
    P2 = Point3D(X=0.0, Y=8000.0, Z=80.0)
    P3 = Point3D(X=12000.0, Y=8000.0, Z=150.0)
    P4 = Point3D(X=12000.0, Y=0.0, Z=200.0)
    P5 = Point3D(X=6000.0, Y=4000.0, Z=-100.0)

    # Create a list of control points
    control_points = [P1, P2, P3, P4, P5]

    # Create the 3D surface from the control points
    surface = Surface3D(control_points)

    # Print surface information
    surface.print_surface_info()

    # Example: Query Z-values at various XY positions on the surface
    test_points = [
        (3000.0, 3000.0),
        (6000.0, 5000.0),
        (9000.0, 6000.0),
        (6000.0, 7000.0),
        (12500.0, 4000.0),  # Outside convex hull
        (3500.0, -200.0),
        (6000, -200.0),
        (6000.0, 8500.0),
    ]

    print("\nInterpolation Results:")
    print("-" * 60)
    for x, y in test_points:
        try:
            z = surface.get_z_at_xy(x, y)
            status = "INSIDE"
        except ValueError:
            z = surface.interpolator(x, y)
            status = "OUTSIDE"
        print(f"Query XY ({x:8.1f}, {y:8.1f}) → Z={z:8.2f} [{status}]")

    print("\n" + "=" * 60)
    print("Surface interpolation completed successfully!")
    print("=" * 60)

    # Visualize the surface with control points and interpolated test points
    print("\nGenerating 3D surface visualization...")
    print("  - Blue triangles: Points inside the convex hull (interpolated)")
    print(
        "  - Orange inverted triangles: Points outside the convex hull (extrapolated)"
    )
    print("  - Viridis colored surface: Interpolated surface inside hull")
    print("  - Gray surface: Extrapolated surface outside hull")
    surface.visualize_surface(
        interpolation_density=30,
        test_points=test_points,
        title="3D Surface Interpolation - Well Surface Example",
        show_extrapolation=True,
    )


if __name__ == "__main__":
    main()
