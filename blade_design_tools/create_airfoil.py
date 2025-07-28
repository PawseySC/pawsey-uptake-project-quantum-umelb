import numpy as np
import matplotlib.pyplot as plt
import yaml # For reading YAML configuration files

# Helper function for Bezier curve calculation (de Casteljau algorithm)
def de_casteljau(t, points):
    """
    Calculates a point on a Bezier curve using the de Casteljau algorithm.
    This algorithm works for any number of control points (any degree).

    Args:
        t (float): Parameter from 0 to 1.
        points (np.array): A Nx2 NumPy array of control points [[x0,y0], ..., [xN-1,yN-1]].

    Returns:
        np.array: The (x, y) coordinates of the point on the curve at parameter t.
    """
    temp_points = np.copy(points)
    n = len(temp_points)

    for r in range(1, n):
        for i in range(n - r):
            temp_points[i] = (1 - t) * temp_points[i] + t * temp_points[i+1]
    return temp_points[0]

# --- Cubic Bezier Functions ---
def compute_cubic_bezier_control_points(P0, t0, P3, t3, alpha, beta):
    """
    Computes the four control points for a cubic Bezier curve given
    start/end points and tangent vectors scaled by alpha/beta.
    P0: Start point
    t0: Tangent vector at P0 (normalized)
    P3: End point
    t3: Tangent vector at P3 (normalized)
    alpha: Scaling factor for t0
    beta: Scaling factor for t3
    """
    t0 = t0 / np.linalg.norm(t0) # Ensure t0 is normalized
    t3 = t3 / np.linalg.norm(t3) # Ensure t3 is normalized

    P1 = P0 + alpha * t0
    P2 = P3 - beta * t3 # Note the subtraction as t3 points from P2 to P3

    return np.array([P0, P1, P2, P3])

def de_casteljau_cubic(t, control_points):
    """
    Cubic Bézier point calculation using De Casteljau's algorithm.
    control_points: A 4x2 NumPy array of control points (P0, P1, P2, P3).
    t: Parameter from 0 to 1.
    """
    P = control_points
    # Linear interpolations for the first level
    A = (1 - t) * P[0] + t * P[1]
    B = (1 - t) * P[1] + t * P[2]
    C = (1 - t) * P[2] + t * P[3]
    # Linear interpolations for the second level
    D = (1 - t) * A + t * B
    E = (1 - t) * B + t * C
    # Linear interpolation for the final point
    point = (1 - t) * D + t * E
    return point
# --- End of Cubic Bezier Functions ---

def load_config(filepath):
    """Loads configuration from a YAML file."""
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_curve_points(curve_params, curve_type, num_points_on_curve):
    """
    Computes the Bezier control points and generates the curve points
    for camber or thickness distributions.
    """
    params = curve_params['control_points_params']
    angles = curve_params

    num_control_points = 5 # Fixed at 5 points (P0, P1, P2, P3, P4)
    control_points = np.zeros((num_control_points, 2), dtype=float)

    # P0 (first point)
    if curve_type == "camber":
        control_points[0] = [0.0, 0.0] # Fixed at (0,0) for camber
    else: # For thickness curves
        control_points[0] = [0.0, angles['le_y_thickness']] # P0.x fixed at 0.0, P0.y from config

    # P4 (last point) - fixed X at 1.0
    if curve_type == "camber":
        stagger_angle_rad = np.deg2rad(angles['stagger_angle_deg'])
        initial_P4_y = np.tan(stagger_angle_rad)
        control_points[num_control_points - 1] = [1.0, initial_P4_y]
    else: # For thickness curves
        control_points[num_control_points - 1] = [1.0, angles['p4_y']] # P4.x fixed at 1.0, P4.y from config

    # Calculate inlet and outlet direction vectors
    inlet_angle_rad = np.deg2rad(angles['inlet_angle_deg'])
    inlet_dir_vec = np.array([np.cos(inlet_angle_rad), np.sin(inlet_angle_rad)])
    norm_inlet = np.linalg.norm(inlet_dir_vec)
    if norm_inlet > 1e-6:
        inlet_dir_vec = inlet_dir_vec / norm_inlet
    else:
        inlet_dir_vec = np.array([0.0, 1.0]) # Default to vertical

    outlet_angle_rad = np.deg2rad(angles['outlet_angle_deg'])
    outlet_dir_vec = np.array([np.cos(outlet_angle_rad), np.sin(outlet_angle_rad)])
    norm_outlet = np.linalg.norm(outlet_dir_vec)
    if norm_outlet > 1e-6:
        outlet_dir_vec = outlet_dir_vec / norm_outlet
    else:
        outlet_dir_vec = np.array([0.0, -1.0]) # Default to vertical

    # P1 (second point) and P3 (second-to-last point)
    P0_ref = control_points[0]
    Pn_1_ref = control_points[num_control_points - 1]

    control_points[1] = P0_ref + params['P1_k_factor'] * inlet_dir_vec
    control_points[num_control_points - 2] = Pn_1_ref + params['P3_k_factor'] * outlet_dir_vec

    # P2 (mid-point)
    control_points[2] = [params['P2_x'], params['P2_y']]

    # Ensure thickness Y-values are non-negative for thickness curves
    if curve_type != "camber":
        control_points[0][1] = np.clip(control_points[0][1], 0.0, np.inf)
        control_points[1][1] = np.clip(control_points[1][1], 0.0, np.inf)
        control_points[2][1] = np.clip(control_points[2][1], 0.0, np.inf)
        control_points[3][1] = np.clip(control_points[3][1], 0.0, np.inf)
        control_points[4][1] = np.clip(control_points[4][1], 0.0, np.inf)


    # Generate points on the Bezier curve
    t_values = np.linspace(0, 1, num_points_on_curve)
    curve_points = np.array([de_casteljau(t, control_points) for t in t_values])

    return control_points, curve_points

def generate_airfoil_contour(camber_curve_points, top_thickness_curve_points,
                             bottom_thickness_curve_points, camber_inlet_angle_deg,
                             num_points_on_arc):
    """
    Generates the full airfoil contour by combining camber and thickness
    distributions with cubic Bezier arcs at LE/TE.
    """
    x_camber = camber_curve_points[:, 0]
    y_camber = camber_curve_points[:, 1]

    # Calculate gradient of the camber line (dy/dx)
    dy_dx_camber = np.gradient(y_camber, x_camber)

    # Explicitly set the leading edge gradient to 0 if camber inlet angle is 0
    le_x_camber = 0.0
    le_idx = np.argmin(np.abs(x_camber - le_x_camber))
    if camber_inlet_angle_deg == 0:
        dy_dx_camber[le_idx] = 0.0

    # Calculate normal vectors to the camber line
    normal_magnitude = np.sqrt(1 + dy_dx_camber**2)
    normal_magnitude[normal_magnitude == 0] = 1e-6 # Avoid division by zero

    nx_top = -dy_dx_camber / normal_magnitude
    ny_top = 1 / normal_magnitude

    nx_bottom = dy_dx_camber / normal_magnitude
    ny_bottom = -1 / normal_magnitude

    top_thickness_vals = top_thickness_curve_points[:, 1]
    bottom_thickness_vals = bottom_thickness_curve_points[:, 1]

    # Calculate initial upper and lower surfaces (before arc blending)
    full_airfoil_top_x = x_camber + top_thickness_vals * nx_top
    full_airfoil_top_y = y_camber + top_thickness_vals * ny_top
    full_airfoil_top_points = np.column_stack((full_airfoil_top_x, full_airfoil_top_y))

    full_airfoil_bottom_x = x_camber + bottom_thickness_vals * nx_bottom
    full_airfoil_bottom_y = y_camber + bottom_thickness_vals * ny_bottom
    full_airfoil_bottom_points = np.column_stack((full_airfoil_bottom_x, full_airfoil_bottom_y))

    # Find the index where x_camber is >= le_x_camber (which is 0.0)
    start_main_body_idx = np.where(x_camber >= le_x_camber)[0][0]

    le_arc_upper = None
    le_arc_lower = None
    te_arc_upper = None
    te_arc_lower = None

    t_vals_arc = np.linspace(0, 1, num_points_on_arc)
    idx_mid_arc = num_points_on_arc // 2

    # --- Cubic Bezier Arc Generation for LE and TE ---
    for idx in [0, -1]: # 0 for Leading Edge, -1 for Trailing Edge
        P0_surf = np.array([full_airfoil_top_x[idx], full_airfoil_top_y[idx]]) # Point on top surface
        P3_surf = np.array([full_airfoil_bottom_x[idx], full_airfoil_bottom_y[idx]]) # Point on bottom surface

        # Tangents at the current upper and lower surface points
        dy_dx_upper_surf = np.gradient(full_airfoil_top_y, full_airfoil_top_x)
        t0_surf = np.array([1.0, dy_dx_upper_surf[idx]]) # Tangent for P0_surf

        dy_dx_lower_surf = np.gradient(full_airfoil_bottom_y, full_airfoil_bottom_x)
        t3_surf = np.array([1.0, dy_dx_lower_surf[idx]]) # Tangent for P3_surf

        # Radius (distance from lower surface point to camber line)
        radius = np.linalg.norm(P3_surf - camber_curve_points[idx])

        dist_best = 9999.9
        fact_best = 0.0
        
        # Iterate to find the best 'fact' (scaling factor for alpha/beta)
        for fact in np.linspace(0.01, 2.0, 50): # Start from small non-zero fact
            alpha = beta = radius * fact
            if idx == 0: # Leading Edge
                fact_alpha = -1 # P1 is 'left/behind' of P0 along tangent for top-to-bottom arc
                fact_beta = 1   # P2 is 'right/ahead' of P3 along tangent for top-to-bottom arc
            else: # Trailing Edge
                fact_alpha = 1  # P1 is 'right/ahead' of P0 along tangent
                fact_beta = -1  # P2 is 'left/behind' of P3 along tangent
            
            control_points_arc = compute_cubic_bezier_control_points(P0_surf, t0_surf, P3_surf, t3_surf, fact_alpha * alpha, fact_beta * beta)
            bezier_points_arc = np.array([de_casteljau_cubic(t, control_points_arc) for t in t_vals_arc])
            
            dist_zero = np.linalg.norm(P3_surf - camber_curve_points[idx])
            dist_bez = np.linalg.norm(bezier_points_arc[idx_mid_arc] - camber_curve_points[idx])
            
            dist_tmp = abs(dist_zero - dist_bez)
            if dist_tmp < dist_best:
                dist_best = dist_tmp
                fact_best = fact

        # Final arc generation with the best factor
        alpha = beta = radius * fact_best
        if idx == 0: # Leading Edge
            fact_alpha = -1
            fact_beta = 1
        else: # Trailing Edge
            fact_alpha = 1
            fact_beta = -1

        control_points_arc = compute_cubic_bezier_control_points(P0_surf, t0_surf, P3_surf, t3_surf, fact_alpha * alpha, fact_beta * beta)
        bezier_points_arc = np.array([de_casteljau_cubic(t, control_points_arc) for t in t_vals_arc])

        # Store arc segments based on LE/TE
        if idx == 0: # LE
            le_arc_upper = np.flip(bezier_points_arc[:idx_mid_arc+1,:], axis=0) # Goes from arc midpoint to top point
            le_arc_lower = bezier_points_arc[idx_mid_arc:,:]                   # Goes from arc midpoint to bottom point
        else: # TE
            te_arc_upper = bezier_points_arc[:idx_mid_arc+1,:] # Goes from top point to arc midpoint
            te_arc_lower = np.flip(bezier_points_arc[idx_mid_arc:,:], axis=0) # Goes from arc midpoint to bottom point, then flipped for consistency

    # --- Concatenate everything for the final airfoil contour ---
    # The main body segments exclude their first and last points, as those are now covered by arcs.
    main_body_upper_segment = full_airfoil_top_points[start_main_body_idx+1:-1]
    main_body_lower_segment = full_airfoil_bottom_points[start_main_body_idx+1:-1]

    # Handle edge case where main body might be too short (e.g., only 2 points, LE and TE)
    if len(full_airfoil_top_points) <= 2:
        main_body_upper_segment = np.array([])
    if len(full_airfoil_bottom_points) <= 2:
        main_body_lower_segment = np.array([])


    # Concatenate upper surface points: LE arc (mid to top) -> main upper body -> TE arc (top to mid)
    final_x_upper = np.concatenate((le_arc_upper[:,0], main_body_upper_segment[:,0], te_arc_upper[:,0]))
    final_y_upper = np.concatenate((le_arc_upper[:,1], main_body_upper_segment[:,1], te_arc_upper[:,1]))
    airfoil_top_points_export = np.column_stack((final_x_upper, final_y_upper))

    # Concatenate lower surface points: LE arc (mid to bottom) -> main lower body -> TE arc (bottom to mid)
    final_x_lower = np.concatenate((le_arc_lower[:,0], main_body_lower_segment[:,0], te_arc_lower[:,0]))
    final_y_lower = np.concatenate((le_arc_lower[:,1], main_body_lower_segment[:,1], te_arc_lower[:,1]))
    airfoil_bottom_points_export = np.column_stack((final_x_lower, final_y_lower))

    return airfoil_top_points_export, airfoil_bottom_points_export

def plot_airfoil(camber_points, airfoil_top_points, airfoil_bottom_points, plot_title):
    """Plots the generated airfoil, camber line, and thickness distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_title(plot_title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot camber line
    ax.plot(camber_points[:, 0], camber_points[:, 1], 'k--', linewidth=1, label='Camber Line')

    # Plot airfoil contour (top surface + reversed bottom surface)
    full_airfoil_contour_x = np.concatenate((airfoil_top_points[:, 0], airfoil_bottom_points[:, 0][::-1]))
    full_airfoil_contour_y = np.concatenate((airfoil_top_points[:, 1], airfoil_bottom_points[:, 1][::-1]))
    ax.plot(full_airfoil_contour_x, full_airfoil_contour_y, '-', color='black', linewidth=2, label='Airfoil Contour')

    ax.legend()
    plt.show()

def write_points_to_file(data_points, filename, header_comment):
    """Writes (x, y) data points to a text file."""
    if data_points is None or len(data_points) == 0:
        print(f"No data points to write for {filename}.")
        return
    try:
        with open(filename, 'w') as f:
            f.write(f"# {header_comment}\n")
            f.write("# X-coordinate\tY-coordinate\n")
            for x, y in data_points:
                f.write(f"{x:.6f}\t{y:.6f}\n")
        print(f"Successfully wrote data to {filename}")
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")

def write_airfoil_dat(airfoil_top_points, airfoil_bottom_points, camber_points,
                      output_filename, chord_length_for_export):
    """
    Writes the combined airfoil contour to an .dat file,
    after shifting the LE to (0,0) and rescaling to chord_length_for_export.
    """
    if airfoil_top_points is None or airfoil_bottom_points is None or camber_points is None:
        print("Airfoil data not generated yet for export.")
        return

    # Make copies to avoid modifying the original data
    x_upper_transformed = np.copy(airfoil_top_points[:, 0])
    y_upper_transformed = np.copy(airfoil_top_points[:, 1])
    x_lower_transformed = np.copy(airfoil_bottom_points[:, 0])
    y_lower_transformed = np.copy(airfoil_bottom_points[:, 1])

    # Shift: Make the leading edge of the camber line (0,0)
    # The camber line points are always from X=0 to X=1 in the normalized system.
    dx = camber_points[0,0]
    dy = camber_points[0,1]

    x_upper_transformed -= dx
    y_upper_transformed -= dy
    x_lower_transformed -= dx
    y_lower_transformed -= dy

    # Rescale: Based on the chord length of the camber line, which is 1.0 in normalized coords
    current_chord_length = camber_points[-1,0] - camber_points[0,0]
    if current_chord_length < 1e-6: # Avoid division by zero if chord is effectively zero
        print("Warning: Current chord length is too small for rescaling. Using 1.0.")
        scale = 1.0 # No scaling
    else:
        scale = chord_length_for_export / current_chord_length

    x_upper_transformed /= scale
    y_upper_transformed /= scale
    x_lower_transformed /= scale
    y_lower_transformed /= scale

    try:
        with open(output_filename, "w") as io_file:
            io_file.writelines(f"{chord_length_for_export:.3f}\n") # Write chord length first
            n_upper = x_upper_transformed.shape[0]
            n_lower = x_lower_transformed.shape[0]
            io_file.writelines(f"{n_upper}\n") # Number of upper points
            for n in range(n_upper):
                io_file.writelines(f"{x_upper_transformed[n]:.15f} {y_upper_transformed[n]:.15f} \n")
            io_file.writelines(f"{n_lower}\n") # Number of lower points
            for n in range(n_lower):
                io_file.writelines(f"{x_lower_transformed[n]:.15f} {y_lower_transformed[n]:.15f} \n")
        print("Wrote:", output_filename)
    except Exception as e:
        print(f"Error writing to file {output_filename}: {e}")

def generate_shape(config_filepath="airfoil_config.yaml"):
    
    try:
        config = load_config(config_filepath)

        # 1. Compute Camber Line
        camber_control_points, camber_curve_points = compute_curve_points(
            config['camber_line'], "camber", config['output_settings']['num_points_on_curve']
        )
        
        # 2. Compute Top Thickness Distribution
        top_thickness_control_points, top_thickness_curve_points = compute_curve_points(
            config['top_thickness'], "top_thickness", config['output_settings']['num_points_on_curve']
        )

        # 3. Compute Bottom Thickness Distribution
        bottom_thickness_control_points, bottom_thickness_curve_points = compute_curve_points(
            config['bottom_thickness'], "bottom_thickness", config['output_settings']['num_points_on_curve']
        )

        # 4. Generate Full Airfoil Contour
        airfoil_top_points, airfoil_bottom_points = generate_airfoil_contour(
            camber_curve_points,
            top_thickness_curve_points,
            bottom_thickness_curve_points,
            config['camber_line']['inlet_angle_deg'],
            config['output_settings']['num_points_on_arc']
        )

        # 5. Plot the Airfoil
        plot_airfoil(camber_curve_points, airfoil_top_points, airfoil_bottom_points,
                     config['output_settings']['plot_title'])

        # 6. Write output files
        if config['output_settings']['output_camber_filename'] != "None":
            write_points_to_file(camber_curve_points, config['output_settings']['output_camber_filename'], "Camber Line Coordinates")
        if config['output_settings']['output_top_thickness_filename'] != "None":
            write_points_to_file(top_thickness_curve_points, config['output_settings']['output_top_thickness_filename'], "Top Thickness Distribution Coordinates")
        if config['output_settings']['output_bottom_thickness_filename'] != "None":
            write_points_to_file(bottom_thickness_curve_points, config['output_settings']['output_bottom_thickness_filename'], "Bottom Thickness Distribution Coordinates")
        
        if config['output_settings']['output_airfoil_filename'] != "None":
            write_airfoil_dat(airfoil_top_points, airfoil_bottom_points, camber_curve_points,
                            config['output_settings']['output_airfoil_filename'],
                            config['output_settings']['chord_length_for_export'])

    except FileNotFoundError:
        print(f"Error: Configuration file '{config_filepath}' not found.")
        print("Please ensure 'airfoil_config.yaml' is in the same directory as the script.")
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        print("Please check 'airfoil_config.yaml' for correct structure and all required parameters.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    return airfoil_top_points, airfoil_bottom_points, camber_curve_points

if __name__ == "__main__":
    generate_shape()
    