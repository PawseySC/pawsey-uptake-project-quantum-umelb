#######################################################################
# SIMPLE SCRIPT TO GENERATE A NACA LIKE AIRFOIL #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

def generate_blade_profile_naca(
    inlet_angle_deg, outlet_angle_deg, stagger_angle_deg,
    t_max_frac, pitch,
    le_thickness_abs=0.0, # Absolute thickness at leading edge (x=0)
    te_thickness_abs=0.0, # Absolute thickness at trailing edge (x=1)
    n_points=1000,plot_me=True
):
    """
    Generates a NACA-like blade profile with controlled leading and trailing edge thickness.

    Args:
        inlet_angle_deg (float): Inlet angle of the camber line in degrees.
        outlet_angle_deg (float): Outlet angle of the camber line in degrees.
        stagger_angle_deg (float): Stagger angle of the blade in degrees.
        t_max_frac (float): Maximum thickness as a fraction of the chord.
        pitch (float): Pitch of the blade (distance between adjacent blades).
        le_thickness_abs (float, optional): Absolute thickness at the leading edge (x=0).
                                            A non-zero value creates a blunt leading edge. Defaults to 0.0.
        te_thickness_abs (float, optional): Absolute thickness at the trailing edge (x=1). Defaults to 0.0.
        n_points (int, optional): Number of points to generate along the chord. Defaults to 300.

    Returns:
        tuple: (x_upper, y_upper, x_lower, y_lower, x_camber, y_camber, throat_distance)
               Coordinates of upper and lower surfaces, camber line, and throat distance.
    """
    # Convert angles from degrees to radians
    inlet_angle = np.radians(inlet_angle_deg)
    outlet_angle = np.radians(outlet_angle_deg)
    stagger_angle = np.radians(stagger_angle_deg)

    # Define leading edge (LE) and trailing edge (TE) coordinates for the camber line
    x_le, y_le = 0.0, 0.0
    x_te, y_te = 1.0, np.tan(stagger_angle)

    # Create a cubic Hermite spline for the camber line
    # This ensures smooth transitions based on inlet and outlet angles.
    s = np.array([0.0, 1.0])
    y_camber = np.array([y_le, y_te])
    dydx = np.array([np.tan(inlet_angle), np.tan(outlet_angle)])
    camber_spline = CubicHermiteSpline(s, y_camber, dydx)

    # Generate points along the chord (x-axis)
    x = np.linspace(0, 1, n_points)
    y_camber_vals = camber_spline(x)
    dy_dx = camber_spline.derivative()(x)

    # Calculate normals to the camber line
    # These normals are used to offset the thickness perpendicular to the camber line.
    norm = np.sqrt(1 + dy_dx**2)
    nx = -dy_dx / norm # X-component of normal vector
    ny = 1 / norm       # Y-component of normal vector

    # Standard NACA 4-digit thickness distribution (naturally zero at LE and TE)
    def naca_thickness_base(xn):
        """
        Calculates the normalized thickness for a NACA 4-digit airfoil.
        Returns values between 0 and ~0.126 (max at x~0.3).
        """
        return (
            0.2969 * np.sqrt(xn)
            - 0.1260 * xn
            - 0.3516 * xn**2
            + 0.2843 * xn**3
            - 0.085 * xn**4
        )

    # Calculate the base thickness, scaled by t_max_frac
    thickness_base = 5 * t_max_frac * naca_thickness_base(x)

    # Create a cubic polynomial for smoothly adding leading/trailing edge thickness.
    # The polynomial P(x) is designed such that:
    # P(0) = le_thickness_abs (desired thickness at LE)
    # P(1) = te_thickness_abs (desired thickness at TE)
    # P'(0) = 0 (zero derivative at LE for smooth transition)
    # P'(1) = 0 (zero derivative at TE for smooth transition)
    # The coefficients A, B, C, D are derived from these conditions.
    A = 2 * (le_thickness_abs - te_thickness_abs)
    B = -3 * (le_thickness_abs - te_thickness_abs)
    C = 0
    D = le_thickness_abs

    thickness_correction = A * x**3 + B * x**2 + C * x + D

    # Combine the base NACA thickness with the polynomial correction
    thickness = thickness_base + thickness_correction

    # Ensure thickness values are non-negative
    thickness[thickness < 0] = 0

    # Calculate upper and lower surface coordinates by offsetting from the camber line
    x_upper = x + 0.5 * thickness * nx
    y_upper = y_camber_vals + 0.5 * thickness * ny
    x_lower = x - 0.5 * thickness * nx
    y_lower = y_camber_vals - 0.5 * thickness * ny

    # Trailing Edge Arc Closure:
    # The original code uses an arc to close the trailing edge.
    # The radius of this arc is now based on the calculated thickness at the trailing edge.
    # If te_thickness_abs is 0, a small default radius is used to prevent issues.
    if thickness[-1] > 0:
        radius = thickness[-1] / 2
    else:
        radius = 0.001 # Small default radius if TE thickness is effectively zero

    theta = np.linspace(0, 2 * np.pi, 100) # Full circle for finding intersection points
    x_arc = x[-1] + radius * np.cos(theta)
    y_arc = y_camber_vals[-1] + radius * np.sin(theta)

    # Calculate and plot throat (minimum passage width)
    throat_search_length = 10.0 # Length to search for throat
    x_throat_line = np.linspace(x_lower[-1], x_lower[-1] - nx[-1] * throat_search_length, n_points)
    y_throat_line = np.linspace(y_lower[-1], y_lower[-1] - ny[-1] * throat_search_length, n_points)

    min_dist_sq = np.inf
    x_throat_point, y_throat_point = 0, 0 # Initialize variables

    # Find the closest point on the upper surface of the adjacent blade to the lower surface of the current blade
    for x_t, y_t in zip(x_throat_line, y_throat_line):
        for x_u, y_u in zip(x_upper, y_upper - pitch): # Upper surface of adjacent blade
            d_sq = (x_t - x_u)**2 + (y_t - y_u)**2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                x_throat_point, y_throat_point = x_u, y_u

    throat_distance = np.linalg.norm([[x_throat_point - x_lower[-1]], [y_throat_point - y_lower[-1]]])

    if plot_me:
        # --- Plotting for visualization ---
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, y_camber_vals, 'k--', label='Camber Line') # Plot camber line

        # Plot upper and lower surfaces, shifted by pitch for visualization of multiple blades
        for i in range(2):
            ax.plot(x_lower, y_lower - i * pitch, 'b', label='Lower Surface' if i == 0 else "")
            ax.plot(x_upper, y_upper - i * pitch, 'r', label='Upper Surface' if i == 0 else "")
        ax.plot([x_lower[-1], x_throat_point], [y_lower[-1], y_throat_point], color='grey', linestyle=':', label='Throat Line')
        # ax.scatter(x_throat_point, y_throat_point, color='purple', zorder=5, label='Throat Point')

    # Find intersection points of the trailing edge arc with the upper and lower surfaces
    dist_squared_lower = (x_arc - x_lower[-1])**2 + (y_arc - y_lower[-1])**2
    i_closest_lower = np.argmin(dist_squared_lower)
    dist_squared_upper = (x_arc - x_upper[-1])**2 + (y_arc - y_upper[-1])**2
    i_closest_upper = np.argmin(dist_squared_upper)

    # Concatenate the main profile with the trailing edge arc to close the profile
    # The arc points are ordered to correctly connect the upper and lower surfaces.
    x_arc_outer = np.concatenate((x_arc[:i_closest_upper][::-1], x_arc[i_closest_lower:-1][::-1]))
    y_arc_outer = np.concatenate((y_arc[:i_closest_upper][::-1], y_arc[i_closest_lower:-1][::-1]))
    n_add = x_arc_outer.shape[0]

    te_idx = n_add // 2 # Index for the point on the arc that aligns with the camber line

    # Append the arc points to the profile coordinates
    x = np.append(x, x_arc_outer[te_idx])
    y_camber_vals = np.append(y_camber_vals, y_arc_outer[te_idx])
    x_upper = np.concatenate((x_upper, x_arc_outer[:n_add//2 + 1]))
    y_upper = np.concatenate((y_upper, y_arc_outer[:n_add//2 + 1]))
    x_lower = np.concatenate((x_lower, np.flip(x_arc_outer[n_add//2:])))
    y_lower = np.concatenate((y_lower, np.flip(y_arc_outer[n_add//2:])))

    # Re-scale the profile to ensure the chord length remains 1.0 after adding the TE arc.
    # The scale factor is the x-coordinate of the new trailing edge point.
    scale_factor = x_arc_outer[te_idx]
    if scale_factor == 0: # Prevent division by zero if TE arc is problematic
        scale_factor = 1.0

    # normalize such that last camber x value is at 1.0 exactly.
    x_upper /= scale_factor
    y_upper /= scale_factor
    x_lower /= scale_factor
    y_lower /= scale_factor
    x /= scale_factor
    y_camber_vals /= scale_factor
    throat_distance /= scale_factor

    if plot_me:
        for i in range(2):
            ax.plot(x_lower, y_lower - i * pitch, 'grey', alpha=0.5, label='Final (rescaled) blade' if i == 0 else "")
            ax.plot(x_upper, y_upper - i * pitch, 'grey', alpha=0.5, label=None)
        ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
        ax.set_xlabel('X/Axial Chord')
        ax.set_ylabel('Y/Axial Chord')
        ax.legend()
        ax.grid(True)
        fname_fig = 'blade.png'
        plt.show()
        # plt.savefig(fname_fig,dpi=300)
        # print('Exported:',fname_fig)
        # --- End Plotting ---

    return x_upper, y_upper, x_lower, y_lower, x, y_camber_vals, throat_distance

if __name__ == '__main__':

    x_upper, y_upper, \
    x_lower, y_lower, \
    x_camber, y_camber, \
    throat = \
    generate_blade_profile_naca(
        inlet_angle_deg=40,
        outlet_angle_deg=-40,
        stagger_angle_deg=-15,
        t_max_frac=0.175,
        pitch=0.6,
        le_thickness_abs=0.0, # Sharp LE
        te_thickness_abs=0.00  # Sharp TE
    )