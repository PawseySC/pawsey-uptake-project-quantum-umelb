import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    # Create a copy of the points to avoid modifying the original array
    # during the recursive reduction process.
    temp_points = np.copy(points)
    n = len(temp_points)

    # The de Casteljau algorithm iteratively interpolates between points.
    # For a curve of degree (n-1), we perform (n-1) steps.
    for r in range(1, n):
        for i in range(n - r):
            # Linear interpolation: P_i^r = (1-t) * P_i^(r-1) + t * P_{i+1}^(r-1)
            temp_points[i] = (1 - t) * temp_points[i] + t * temp_points[i+1]
    # The final point left in temp_points[0] is the point on the Bezier curve.
    return temp_points[0]

# --- New Cubic Bezier Functions from your snippet ---
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
# --- End of New Cubic Bezier Functions ---


class CurveEditorWindow:
    """
    A Tkinter Toplevel window for editing a Bezier curve with specific constraints.
    Can be configured for camber or thickness distribution editing.
    """
    def __init__(self, master, title, curve_type, update_main_plot_callback):
        """
        Initializes the CurveEditorWindow.

        Args:
            master: The parent Tkinter window (MainApplication's root).
            title (str): The title for this window.
            curve_type (str): "Camber", "Top Thickness", or "Bottom Thickness".
            update_main_plot_callback (callable): A function to call in MainApplication
                                                  when this curve's data changes.
        """
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title(title)
        self.window.geometry("800x700")
        self.curve_type = curve_type
        self.update_main_plot_callback = update_main_plot_callback

        self.num_points = 5 # Fixed at 5 points (P0, P1, P2, P3, P4)

        # Initialize StringVars first to ensure they exist before being accessed
        self.p1_y_var = tk.StringVar()
        self.p2_y_var = tk.StringVar()
        self.p3_y_var = tk.StringVar()
        self.p2_x_var = tk.StringVar() # P2 X is always present

        self.inlet_angle_var = tk.StringVar() # Now for both Camber and Thickness
        self.outlet_angle_var = tk.StringVar() # Now for both Camber and Thickness

        if self.curve_type == "Camber":
            self.stagger_angle_var = tk.StringVar()
        else: # For Top/Bottom Thickness
            self.p4_y_var = tk.StringVar() # P4 Y-coordinate for thickness curves
            self.le_y_thickness_var = tk.StringVar() # Changed: LE Y-coordinate for thickness curves

        # Define fixed y-level for P0.
        self.fixed_y_level_P0 = 0.0

        # Initial angles/values based on user request
        if self.curve_type == "Camber":
            self.angle_inlet_deg = 45
            self.angle_outlet_deg = -50
            self.stagger_angle_deg = -25 # Initial stagger angle for Camber
        else: # For Top/Bottom Thickness
            self.initial_p4_y_thickness = 0.01 # Initial P4 Y for thickness
            self.le_y_thickness = 0.02 # Initial LE Y for thickness (small positive value for bluntness)
            self.angle_inlet_deg = 20 # Default vertical tangent for thickness LE
            self.angle_outlet_deg = -0 # Default vertical tangent for thickness TE

        # Initialize control points based on curve type and constraints
        self.control_points = np.zeros((self.num_points, 2), dtype=float)

        # P0 (first point)
        if self.curve_type == "Camber":
            self.control_points[0] = [0.0, self.fixed_y_level_P0]
        else: # For thickness curves, P0.x is fixed at 0.0, P0.y is controlled by le_y_thickness
            self.control_points[0] = [0.0, self.le_y_thickness]

        # P4 (last point) - fixed X at 1.0
        if self.curve_type == "Camber":
            initial_P4_y = np.tan(np.deg2rad(self.stagger_angle_deg))
            self.control_points[self.num_points - 1] = [1.0, initial_P4_y]
        else: # For thickness curves, P4's Y is now controllable, X is fixed at 1.0
            self.control_points[self.num_points - 1] = [1.0, self.initial_p4_y_thickness]

        # Calculate initial direction vectors for P1 and P3
        self._update_direction_vectors()

        # P1 (second point) and P3 (second-to-last point) initialization
        P0 = self.control_points[0]
        Pn_1 = self.control_points[self.num_points - 1]

        # Initial k factors for P1 and P3
        initial_k_P1 = 0.15 # Distance along inlet tangent
        initial_k_P3 = -0.25 # Distance along outlet tangent (negative to point inwards)

        self.control_points[1] = P0 + initial_k_P1 * self.inlet_dir_vec
        self.control_points[self.num_points - 2] = Pn_1 + initial_k_P3 * self.outlet_dir_vec

        # Ensure thickness Y-values are non-negative for thickness curves
        if self.curve_type != "Camber":
            self.control_points[1][1] = np.clip(self.control_points[1][1], 0.0, np.inf)
            self.control_points[self.num_points - 2][1] = np.clip(self.control_points[self.num_points - 2][1], 0.0, np.inf)


        # P2 (mid-point) - free to move
        self.control_points[2] = [0.5, 0.05 if "Thickness" in self.curve_type else 0.05] # Initial Y slightly positive for thickness
        if "Thickness" in self.curve_type: # Ensure positive Y for thickness
            self.control_points[2][1] = np.clip(self.control_points[2][1], 0.0, np.inf)


        self.selected_point_index = None # Index of the currently dragged point

        # Setup Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.curve_type} (Degree {self.num_points - 1})")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.grid(True, linestyle='--', alpha=0.7)

        # --- Input Fields for Y-coordinates and Angles ---
        self.input_frame = tk.Frame(self.window, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Row 0: Y-coordinates
        tk.Label(self.input_frame, text="P1 Y:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.p1_y_entry = tk.Entry(self.input_frame, textvariable=self.p1_y_var, width=10)
        self.p1_y_entry.grid(row=0, column=1, padx=5, pady=2, sticky='ew')

        tk.Label(self.input_frame, text="P2 Y:").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.p2_y_entry = tk.Entry(self.input_frame, textvariable=self.p2_y_var, width=10)
        self.p2_y_entry.grid(row=0, column=3, padx=5, pady=2, sticky='ew')

        tk.Label(self.input_frame, text="P3 Y:").grid(row=0, column=4, padx=5, pady=2, sticky='w')
        self.p3_y_entry = tk.Entry(self.input_frame, textvariable=self.p3_y_var, width=10)
        self.p3_y_entry.grid(row=0, column=5, padx=5, pady=2, sticky='ew')

        # Row 1: Angles (Camber and Thickness) and P2 X
        tk.Label(self.input_frame, text="Inlet Angle (deg):").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.inlet_angle_entry = tk.Entry(self.input_frame, textvariable=self.inlet_angle_var, width=10)
        self.inlet_angle_entry.grid(row=1, column=1, padx=5, pady=2, sticky='ew')

        tk.Label(self.input_frame, text="Outlet Angle (deg):").grid(row=1, column=2, padx=5, pady=2, sticky='w')
        self.outlet_angle_entry = tk.Entry(self.input_frame, textvariable=self.outlet_angle_var, width=10)
        self.outlet_angle_entry.grid(row=1, column=3, padx=5, pady=2, sticky='ew')

        tk.Label(self.input_frame, text="P2 X:").grid(row=1, column=4, padx=5, pady=2, sticky='w')
        self.p2_x_entry = tk.Entry(self.input_frame, textvariable=self.p2_x_var, width=10)
        self.p2_x_entry.grid(row=1, column=5, padx=5, pady=2, sticky='ew')

        # Row 2: Stagger Angle (for Camber) OR P4 Y & LE Y (for Thickness) and Button
        if self.curve_type == "Camber":
            tk.Label(self.input_frame, text="Stagger Angle (deg):").grid(row=2, column=0, padx=5, pady=2, sticky='w')
            self.stagger_angle_entry = tk.Entry(self.input_frame, textvariable=self.stagger_angle_var, width=10)
            self.stagger_angle_entry.grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        else: # For Top/Bottom Thickness
            tk.Label(self.input_frame, text="LE Y:").grid(row=2, column=0, padx=5, pady=2, sticky='w')
            self.le_y_thickness_entry = tk.Entry(self.input_frame, textvariable=self.le_y_thickness_var, width=10)
            self.le_y_thickness_entry.grid(row=2, column=1, padx=5, pady=2, sticky='ew')

            tk.Label(self.input_frame, text="P4 Y:").grid(row=2, column=2, padx=5, pady=2, sticky='w')
            self.p4_y_entry = tk.Entry(self.input_frame, textvariable=self.p4_y_var, width=10)
            self.p4_y_entry.grid(row=2, column=3, padx=5, pady=2, sticky='ew')

        # Set All Coordinates Button
        self.set_coords_button = tk.Button(self.input_frame, text="Set All Coordinates", command=self.update_points_from_inputs)
        self.set_coords_button.grid(row=2, column=6, padx=10, pady=2, sticky='e')

        # Configure columns to expand evenly
        for i in range(7):
            self.input_frame.grid_columnconfigure(i, weight=1)

        # Embed Matplotlib figure into Tkinter canvas
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bind mouse events to the Matplotlib canvas
        self.canvas_widget.mpl_connect('button_press_event', self.on_button_press)
        self.canvas_widget.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.canvas_widget.mpl_connect('button_release_event', self.on_button_release)

        # Initial curve drawing and input field population will happen after MainApplication
        # has fully initialized all windows.

    def _update_direction_vectors(self):
        """
        Recalculates inlet and outlet direction vectors based on current angle settings.
        """
        inlet_angle_rad = np.deg2rad(self.angle_inlet_deg)
        self.inlet_dir_vec = np.array([np.cos(inlet_angle_rad), np.sin(inlet_angle_rad)])
        # Normalize to unit vector, handle zero vector case
        norm_inlet = np.linalg.norm(self.inlet_dir_vec)
        if norm_inlet > 1e-6:
            self.inlet_dir_vec = self.inlet_dir_vec / norm_inlet
        else: # Default to vertical if angle is undefined or very close to 0/180
            self.inlet_dir_vec = np.array([0.0, 1.0])


        outlet_angle_rad = np.deg2rad(self.angle_outlet_deg)
        self.outlet_dir_vec = np.array([np.cos(outlet_angle_rad), np.sin(outlet_angle_rad)])
        # Normalize to unit vector, handle zero vector case
        norm_outlet = np.linalg.norm(self.outlet_dir_vec)
        if norm_outlet > 1e-6:
            self.outlet_dir_vec = self.outlet_dir_vec / norm_outlet
        else: # Default to vertical if angle is undefined or very close to 0/180
            self.outlet_dir_vec = np.array([0.0, -1.0]) # Pointing downwards


    def draw_curve(self, update_from_drag=False):
        """
        Clears the Matplotlib axes and redraws the control points,
        the control polygon, and the Bezier curve.
        `update_from_drag` is used to prevent input fields from being updated
        during continuous dragging, as it can be slow.
        """
        self.ax.clear() # Clear existing plot elements

        # Set axis limits and title again after clearing
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.curve_type} (Degree {self.num_points - 1})")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.grid(True, linestyle='--', alpha=0.7)

        # Draw all control points
        for i, point in enumerate(self.control_points):
            color = 'red' # Default color
            if i == 0:
                color = 'green' # Fixed P0
            elif i == self.num_points - 1:
                color = 'green' # Fixed P4
            elif i == 1 or i == self.num_points - 2:
                color = 'orange' # Sloped/constrained points (P1, P3)
            else:
                color = 'blue' # Free points (P2)

            label = f'P{i}'
            self.ax.plot(point[0], point[1], 'o', markersize=12, color=color, label=label)

        # Draw control polygon (lines connecting control points)
        self.ax.plot(self.control_points[:, 0], self.control_points[:, 1],
                     '--', color='gray', linewidth=1, label='Control Polygon')

        # Generate points for the Bezier curve using de Casteljau algorithm
        num_segments = 100 # Number of segments to draw the curve smoothly
        t_values = np.linspace(0, 1, num_segments)
        bezier_points = np.array([de_casteljau(t, self.control_points) for t in t_values])

        # Draw the Bezier curve
        self.ax.plot(bezier_points[:, 0], bezier_points[:, 1],
                     '-', color='purple', linewidth=1, label='Bezier Curve')

        # Update input fields to reflect current point positions, but not during drag
        if not update_from_drag:
            # P1 and P3 Y values are now effectively 'k' factors scaled by their direction vectors
            P0 = self.control_points[0]
            Pn_1 = self.control_points[self.num_points - 1]
            
            # Calculate k_P1 and k_P3 from current P1 and P3 positions
            if np.linalg.norm(self.inlet_dir_vec) > 1e-6:
                k_P1 = np.dot(self.control_points[1] - P0, self.inlet_dir_vec) / np.linalg.norm(self.inlet_dir_vec)**2
            else:
                k_P1 = 0.0 # Should not happen with well-defined angles
            
            if np.linalg.norm(self.outlet_dir_vec) > 1e-6:
                k_P3 = np.dot(self.control_points[self.num_points - 2] - Pn_1, self.outlet_dir_vec) / np.linalg.norm(self.outlet_dir_vec)**2
            else:
                k_P3 = 0.0

            self.p1_y_var.set(f"{k_P1:.2f}") # P1 Y now represents k_P1
            self.p2_y_var.set(f"{self.control_points[2][1]:.2f}")
            self.p3_y_var.set(f"{k_P3:.2f}") # P3 Y now represents k_P3
            self.p2_x_var.set(f"{self.control_points[2][0]:.2f}")
            
            self.inlet_angle_var.set(f"{self.angle_inlet_deg:.2f}")
            self.outlet_angle_var.set(f"{self.angle_outlet_deg:.2f}")

            if self.curve_type == "Camber":
                self.stagger_angle_var.set(f"{self.stagger_angle_deg:.2f}")
            else:
                self.p4_y_var.set(f"{self.control_points[self.num_points - 1][1]:.2f}") # Update P4 Y for thickness
                self.le_y_thickness_var.set(f"{self.control_points[0][1]:.4f}") # Update LE Y for thickness

        self.canvas_widget.draw() # Redraw the Matplotlib canvas
        self.update_main_plot_callback() # Notify main app to redraw airfoil

    def update_points_from_inputs(self):
        """
        Updates control point Y-coordinates, angles, and P2 X based on input field values.
        Calculates X-coordinates for P1 and P3 based on their slopes.
        Updates P4's Y based on stagger angle (for Camber) or direct input (for Thickness).
        """
        try:
            # Update angles first, as they affect point calculations
            self.angle_inlet_deg = float(self.inlet_angle_var.get())
            self.angle_outlet_deg = float(self.outlet_angle_var.get())
            self._update_direction_vectors() # Update direction vectors for both camber and thickness

            if self.curve_type == "Camber":
                self.stagger_angle_deg = float(self.stagger_angle_var.get())
                new_P4_y = np.tan(np.deg2rad(self.stagger_angle_deg))
                self.control_points[self.num_points - 1][1] = new_P4_y
            else: # Thickness curves
                self.le_y_thickness = float(self.le_y_thickness_var.get())
                self.control_points[0][1] = np.clip(self.le_y_thickness, 0.0, np.inf) # Update P0.y, clamped
                self.control_points[0][0] = 0.0 # P0.x fixed at 0.0

                new_P4_y = float(self.p4_y_var.get())
                self.control_points[self.num_points - 1][1] = np.clip(new_P4_y, 0.0, np.inf) # Clamp P4 Y to be non-negative
                self.control_points[self.num_points - 1][0] = 1.0 # P4.x fixed at 1.0


            P0 = self.control_points[0]
            Pn_1 = self.control_points[self.num_points - 1]

            # P1 Update (Y-coordinate input now acts as k factor)
            k_P1 = float(self.p1_y_var.get())
            self.control_points[1] = P0 + k_P1 * self.inlet_dir_vec
            if self.curve_type != "Camber": # Clamp Y to be non-negative for thickness
                self.control_points[1][1] = np.clip(self.control_points[1][1], 0.0, np.inf)

            # P2 Update (Y-coordinate and X-coordinate input)
            new_P2_y = float(self.p2_y_var.get())
            new_P2_x = float(self.p2_x_var.get())
            if self.curve_type != "Camber": # Clamp Y to be non-negative for thickness
                new_P2_y = np.clip(new_P2_y, 0.0, np.inf)
            self.control_points[2] = np.array([new_P2_x, new_P2_y])

            # P3 Update (Y-coordinate input now acts as k factor)
            k_P3 = float(self.p3_y_var.get())
            self.control_points[self.num_points - 2] = Pn_1 + k_P3 * self.outlet_dir_vec
            if self.curve_type != "Camber": # Clamp Y to be non-negative for thickness
                self.control_points[self.num_points - 2][1] = np.clip(self.control_points[self.num_points - 2][1], 0.0, np.inf)

            self.draw_curve()
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            # Optionally, add a message box or visual feedback to the user

    def on_button_press(self, event):
        """
        Event handler for mouse button press.
        Checks if a control point is clicked and stores its index.
        """
        if event.xdata is None or event.ydata is None:
            return # Click was outside the plot area

        mouse_pos = np.array([event.xdata, event.ydata])

        # Check if any control point is clicked within a certain tolerance
        tolerance = 0.03 # Scaled tolerance for 0-1 range
        distances = np.linalg.norm(self.control_points - mouse_pos, axis=1)
        closest_point_index = np.argmin(distances)

        if distances[closest_point_index] < tolerance:
            # Only allow dragging for non-fixed points
            # P0 and P4 are fixed for both camber and thickness
            if closest_point_index == 0 or closest_point_index == self.num_points - 1:
                pass
            else:
                self.selected_point_index = closest_point_index

    def on_mouse_drag(self, event):
        """
        Event handler for mouse drag.
        If a point is selected, updates its position according to constraints and redraws the curve.
        """
        if self.selected_point_index is not None and event.xdata is not None and event.ydata is not None:
            mouse_pos = np.array([event.xdata, event.ydata])

            if self.selected_point_index == 1: # P1 (second point)
                P0 = self.control_points[0]
                # Calculate new k factor based on mouse position projecting onto the inlet_dir_vec
                # Project (mouse_pos - P0) onto inlet_dir_vec
                if np.linalg.norm(self.inlet_dir_vec) > 1e-6:
                    k = np.dot(mouse_pos - P0, self.inlet_dir_vec) / np.linalg.norm(self.inlet_dir_vec)**2
                else:
                    k = 0.0 # Should not happen with well-defined angles

                new_P1 = P0 + k * self.inlet_dir_vec
                if self.curve_type != "Camber": # Clamp Y to be non-negative for thickness
                    new_P1[1] = np.clip(new_P1[1], 0.0, np.inf)
                self.control_points[1] = new_P1
            elif self.selected_point_index == self.num_points - 2: # P3 (second-to-last point)
                Pn_1 = self.control_points[self.num_points - 1]
                # Calculate new k factor based on mouse position projecting onto the outlet_dir_vec
                # Project (mouse_pos - Pn_1) onto outlet_dir_vec
                if np.linalg.norm(self.outlet_dir_vec) > 1e-6:
                    k = np.dot(mouse_pos - Pn_1, self.outlet_dir_vec) / np.linalg.norm(self.outlet_dir_vec)**2
                else:
                    k = 0.0
                
                new_P3 = Pn_1 + k * self.outlet_dir_vec
                if self.curve_type != "Camber": # Clamp Y to be non-negative for thickness
                    new_P3[1] = np.clip(new_P3[1], 0.0, np.inf)
                self.control_points[self.num_points - 2] = new_P3
            else: # P2 (mid-point) is free
                new_P2 = np.array([event.xdata, event.ydata])
                if self.curve_type != "Camber": # Clamp Y to be non-negative for thickness
                    new_P2[1] = np.clip(new_P2[1], 0.0, np.inf)
                self.control_points[self.selected_point_index] = new_P2
            self.draw_curve(update_from_drag=True)

    def on_button_release(self, event):
        """
        Event handler for mouse button release.
        Deselects any active point and redraws to update input fields.
        """
        self.selected_point_index = None
        self.draw_curve(update_from_drag=False) # Redraw to update input fields


class AirfoilPlotWindow:
    """
    A Tkinter Toplevel window that plots the combined airfoil shape
    from camber and thickness distributions.
    """
    def __init__(self, master, camber_points_ref, top_thickness_points_ref, bottom_thickness_points_ref, camber_editor_ref):
        """
        Initializes the AirfoilPlotWindow.

        Args:
            master: The parent Tkinter window (MainApplication's root).
            camber_points_ref (np.array): Reference to the camber curve's control points.
            top_thickness_points_ref (np.array): Reference to the top thickness curve's control points.
            bottom_thickness_points_ref (np.array): Reference to the bottom thickness curve's control points.
            camber_editor_ref: Reference to the Camber CurveEditorWindow instance.
        """
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title("Resulting Airfoil Shape")
        self.window.geometry("800x700") # Increased height to ensure buttons are visible

        self.camber_points_ref = camber_points_ref
        self.top_thickness_points_ref = top_thickness_points_ref
        self.bottom_thickness_points_ref = bottom_thickness_points_ref
        self.camber_editor_ref = camber_editor_ref # Store reference to camber editor

        # Store generated curve points to be accessible by write functions
        self.camber_curve_points = None
        self.airfoil_top_points_export = None
        self.airfoil_bottom_points_export = None
        self.top_thickness_curve_points = None
        self.bottom_thickness_curve_points = None


        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_aspect('equal')
        self.ax.set_title("Airfoil Shape")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.grid(True, linestyle='--', alpha=0.7)

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget.draw()
        # Using grid for better layout control
        self.canvas_widget.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # --- Add Export Buttons ---
        self.export_frame = tk.Frame(self.window, bd=2, relief=tk.GROOVE, padx=10, pady=5)
        # Using grid for better layout control
        self.export_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        tk.Button(self.export_frame, text="Write Top Thickness", command=self._write_top_thickness_to_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.export_frame, text="Write Bottom Thickness", command=self._write_bottom_thickness_to_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.export_frame, text="Write Camber", command=self._write_camber_to_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.export_frame, text="Write Airfoil .dat", command=self._write_airfoil_dat).pack(side=tk.LEFT, padx=5)


        # Configure grid row and column weights for expansion
        self.window.grid_rowconfigure(0, weight=1) # Plot area takes most vertical space
        self.window.grid_rowconfigure(1, weight=0) # Button frame takes minimal vertical space
        self.window.grid_columnconfigure(0, weight=1) # Column takes all horizontal space


        self.update_plot() # Initial plot

    def update_plot(self):
        """
        Recalculates and redraws the airfoil shape based on the current control points
        of the camber and thickness curves, including cubic Bezier arcs for LE/TE.
        """
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title("Airfoil Shape")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.grid(True, linestyle='--', alpha=0.7)

        num_segments = 100 # For the main Bezier curves
        t_values = np.linspace(0, 1, num_segments)

        # Generate points for camber line
        self.camber_curve_points = np.array([de_casteljau(t, self.camber_points_ref) for t in t_values])
        x_camber = self.camber_curve_points[:, 0]
        y_camber = self.camber_curve_points[:, 1]
        self.ax.plot(x_camber, y_camber, '-', color='purple', linewidth=1, label='Camber Line')

        # Calculate gradient of the camber line (dy/dx)
        dy_dx_camber = np.gradient(y_camber, x_camber)

        # Explicitly set the leading edge gradient to 0 if camber inlet angle is 0
        le_x_camber = 0.0
        le_idx = np.argmin(np.abs(x_camber - le_x_camber)) # Find index closest to x=0 on camber
        
        # Access camber_editor from its stored reference
        if self.camber_editor_ref.angle_inlet_deg == 0:
            dy_dx_camber[le_idx] = 0.0


        # Calculate normal vectors to the camber line
        normal_magnitude = np.sqrt(1 + dy_dx_camber**2)
        normal_magnitude[normal_magnitude == 0] = 1e-6 # Avoid division by zero

        nx_top = -dy_dx_camber / normal_magnitude
        ny_top = 1 / normal_magnitude

        nx_bottom = -dy_dx_camber / normal_magnitude
        ny_bottom = 1 / normal_magnitude

        # Generate points for top and bottom thickness distributions
        self.top_thickness_curve_points = np.array([de_casteljau(t, self.top_thickness_points_ref) for t in t_values])
        self.bottom_thickness_curve_points = np.array([de_casteljau(t, self.bottom_thickness_points_ref) for t in t_values])

        top_thickness_vals = self.top_thickness_curve_points[:, 1]
        bottom_thickness_vals = self.bottom_thickness_curve_points[:, 1]

        # Ensure thickness values are non-negative
        top_thickness_vals = np.clip(top_thickness_vals, 0.0, np.inf)
        bottom_thickness_vals = np.clip(bottom_thickness_vals, 0.0, np.inf)

        # Calculate initial upper and lower surfaces (before arc blending)
        x_upper_init = x_camber + top_thickness_vals * nx_top
        y_upper_init = y_camber + top_thickness_vals * ny_top
        x_lower_init = x_camber - bottom_thickness_vals * nx_bottom
        y_lower_init = y_camber - bottom_thickness_vals * ny_bottom

        self.ax.plot(x_lower_init, y_lower_init, 'r--', label='Initial Lower')
        self.ax.plot(x_upper_init, y_upper_init, 'b--', label='Initial Upper')

        # --- Cubic Bezier Arc Generation for LE and TE ---
        le_arc_upper = None
        le_arc_lower = None
        te_arc_upper = None
        te_arc_lower = None

        nt_arc = 100 # Number of points for arc generation
        t_vals_arc = np.linspace(0, 1, nt_arc)
        idx_mid_arc = nt_arc // 2

        for idx in [0, -1]: # 0 for Leading Edge, -1 for Trailing Edge
            P0 = np.array([x_upper_init[idx], y_upper_init[idx]]) # Point on top surface
            P3 = np.array([x_lower_init[idx], y_lower_init[idx]]) # Point on bottom surface

            # Tangents at the current upper and lower surface points
            dy_dx_upper_surf = np.gradient(y_upper_init, x_upper_init)
            t0 = np.array([1.0, dy_dx_upper_surf[idx]]) # Tangent for P0

            dy_dx_lower_surf = np.gradient(y_lower_init, x_lower_init)
            t3 = np.array([1.0, dy_dx_lower_surf[idx]]) # Tangent for P3

            # Radius (distance from lower surface point to camber line)
            # This 'radius' is effectively the half-thickness at the end point
            radius = np.linalg.norm(P3 - self.camber_curve_points[idx])

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
                
                control_points_arc = compute_cubic_bezier_control_points(P0, t0, P3, t3, fact_alpha * alpha, fact_beta * beta)
                bezier_points_arc = np.array([de_casteljau_cubic(t, control_points_arc) for t in t_vals_arc])
                
                # Check distance from arc midpoint to camber line midpoint
                # The goal is for the arc to 'pass through' the camber line at the end
                # or maintain a shape consistent with the thickness distribution.
                # The snippet used P3 (lower surface point) as reference, let's keep that.
                dist_zero = np.linalg.norm(P3 - self.camber_curve_points[idx])
                dist_bez = np.linalg.norm(bezier_points_arc[idx_mid_arc] - self.camber_curve_points[idx])
                
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

            control_points_arc = compute_cubic_bezier_control_points(P0, t0, P3, t3, fact_alpha * alpha, fact_beta * beta)
            bezier_points_arc = np.array([de_casteljau_cubic(t, control_points_arc) for t in t_vals_arc])

            # Store arc segments based on LE/TE
            if idx == 0: # LE
                # bezier_points_arc typically goes from top LE point to bottom LE point
                le_arc_upper = np.flip(bezier_points_arc[:idx_mid_arc+1,:], axis=0) # Goes from arc midpoint to top point
                le_arc_lower = bezier_points_arc[idx_mid_arc:,:]                   # Goes from arc midpoint to bottom point
                # self.ax.plot(le_arc_upper[:,0], le_arc_upper[:,1], 'go-')
                # self.ax.plot(le_arc_lower[:,0], le_arc_lower[:,1], 'bo-')

            else: # TE
                # bezier_points_arc typically goes from top TE point to bottom TE point
                te_arc_upper = bezier_points_arc[:idx_mid_arc+1,:] # Goes from top point to arc midpoint
                te_arc_lower = np.flip(bezier_points_arc[idx_mid_arc:,:], axis=0) # Goes from arc midpoint to bottom point, then flipped for consistency
                # self.ax.plot(te_arc_upper[:,0], te_arc_upper[:,1], 'go-')
                # self.ax.plot(te_arc_lower[:,0], te_arc_lower[:,1], 'bo-')

            # Optional: Plot arc control points
            # self.ax.plot(control_points_arc.T[0], control_points_arc.T[1], 'rx')

        # --- Concatenate everything for the final airfoil contour ---
        # The main body segments exclude their first and last points, as those are now covered by arcs.
        # This prevents duplicate points at the connection interfaces.
        # If the number of points in the main body is less than 3 (i.e., just LE and TE points),
        # then the middle segment is empty.
        
        main_body_upper_segment_points = x_upper_init[1:-1] if len(x_upper_init) > 2 else np.array([])
        main_body_y_upper_segment_points = y_upper_init[1:-1] if len(y_upper_init) > 2 else np.array([])
        
        main_body_lower_segment_points = x_lower_init[1:-1] if len(x_lower_init) > 2 else np.array([])
        main_body_y_lower_segment_points = y_lower_init[1:-1] if len(y_lower_init) > 2 else np.array([])

        # Concatenate upper surface points: LE arc (mid to top) -> main upper body -> TE arc (top to mid)
        final_x_upper = np.concatenate((le_arc_upper[:,0], main_body_upper_segment_points, te_arc_upper[:,0]))
        final_y_upper = np.concatenate((le_arc_upper[:,1], main_body_y_upper_segment_points, te_arc_upper[:,1]))

        # Concatenate lower surface points: LE arc (mid to bottom) -> main lower body -> TE arc (bottom to mid)
        final_x_lower = np.concatenate((le_arc_lower[:,0], main_body_lower_segment_points, te_arc_lower[:,0]))
        final_y_lower = np.concatenate((le_arc_lower[:,1], main_body_y_lower_segment_points, te_arc_lower[:,1]))

        # Store for export (these are in the 0-1 normalized chord space)
        self.airfoil_top_points_export = np.column_stack((final_x_upper, final_y_upper))
        self.airfoil_bottom_points_export = np.column_stack((final_x_lower, final_y_lower))

        # Combine for plotting the full airfoil contour (top from LE to TE, then bottom from TE to LE)
        # Note: final_x_lower is already from LE to TE. For concatenation, we want it reversed to go TE to LE.
        full_airfoil_contour_x = np.concatenate((final_x_upper, final_x_lower[::-1]))
        full_airfoil_contour_y = np.concatenate((final_y_upper, final_y_lower[::-1]))
        
        self.ax.plot(full_airfoil_contour_x, full_airfoil_contour_y, '-', color='black', linewidth=1, label='Airfoil Contour')

        # self.ax.legend()
        self.canvas_widget.draw()

    def _write_to_file(self, filename, data_points, header_comment):
        """
        Writes (x, y) data points to a text file.
        """
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

    def _write_top_thickness_to_file(self):
        # Export the top thickness curve points, not the airfoil top surface points
        if self.top_thickness_curve_points is not None:
            self._write_to_file("airfoil_top_thickness.txt", self.top_thickness_curve_points, "Airfoil Top Thickness Distribution Coordinates")
        else:
            print("Top thickness data not available.")

    def _write_bottom_thickness_to_file(self):
        # Export the bottom thickness curve points, not the airfoil bottom surface points
        if self.bottom_thickness_curve_points is not None:
            self._write_to_file("airfoil_bottom_thickness.txt", self.bottom_thickness_curve_points, "Airfoil Bottom Thickness Distribution Coordinates")
        else:
            print("Bottom thickness data not available.")

    def _write_camber_to_file(self):
        if self.camber_curve_points is not None:
            self._write_to_file("airfoil_camber_line.txt", self.camber_curve_points, "Airfoil Camber Line Coordinates")
        else:
            print("Camber line data not available.")

    def _write_airfoil_dat(self, cax_dim=50.0):
        """
        Writes the combined airfoil contour to an .dat file,
        after shifting the LE to (0,0) and rescaling to cax_dim.
        """
        if self.airfoil_top_points_export is None or self.airfoil_bottom_points_export is None:
            print("Airfoil data not generated yet for export.")
            return

        # Make copies to avoid modifying the plotting data
        x_upper_transformed = np.copy(self.airfoil_top_points_export[:, 0])
        y_upper_transformed = np.copy(self.airfoil_top_points_export[:, 1])
        x_lower_transformed = np.copy(self.airfoil_bottom_points_export[:, 0])
        y_lower_transformed = np.copy(self.airfoil_bottom_points_export[:, 1])

        # Shift: Make the leading edge of the camber line (0,0)
        # The camber line points are always from X=0 to X=1 in the normalized system.
        # So, the shift will bring the initial X=0 point of the camber to 0.
        dx = self.camber_curve_points[0,0]
        dy = self.camber_curve_points[0,1]

        x_upper_transformed -= dx
        y_upper_transformed -= dy
        x_lower_transformed -= dx
        y_lower_transformed -= dy

        # Rescale: Based on the chord length of the camber line, which is 1.0 in normalized coords
        current_chord_length = self.camber_curve_points[-1,0] - self.camber_curve_points[0,0]
        if current_chord_length < 1e-6: # Avoid division by zero if chord is effectively zero
            print("Warning: Current chord length is too small for rescaling.")
            scale = 1.0 # No scaling
        else:
            scale = cax_dim / current_chord_length

        x_upper_transformed /= scale
        y_upper_transformed /= scale
        x_lower_transformed /= scale
        y_lower_transformed /= scale

        fname_out = "./airfoil.dat"
        try:
            with open(fname_out, "w") as io_file:
                io_file.writelines(f"{cax_dim:.3f}\n") # Write chord length first
                n_upper = x_upper_transformed.shape[0]
                n_lower = x_lower_transformed.shape[0]
                io_file.writelines(f"{n_upper}\n") # Number of upper points
                for n in range(n_upper):
                    io_file.writelines(f"{x_upper_transformed[n]:.15f} {y_upper_transformed[n]:.15f} \n")
                io_file.writelines(f"{n_lower}\n") # Number of lower points
                for n in range(n_lower):
                    io_file.writelines(f"{x_lower_transformed[n]:.15f} {y_lower_transformed[n]:.15f} \n")
            print("Wrote:", fname_out)
        except Exception as e:
            print(f"Error writing to file {fname_out}: {e}")


class MainApplication:
    """
    The main application class that orchestrates the creation and synchronization
    of all airfoil design windows.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw() # Hide the main Tkinter root window

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate dimensions for each window (2x2 grid)
        window_width = screen_width // 2
        window_height = screen_height // 2

        # Create the curve editor windows
        self.camber_editor = CurveEditorWindow(self.root, "Camber Line Editor", "Camber", self._update_airfoil_plot)
        self.top_thickness_editor = CurveEditorWindow(self.root, "Top Surface Thickness Editor", "Top Thickness", self._update_airfoil_plot)
        self.bottom_thickness_editor = CurveEditorWindow(self.root, "Bottom Surface Thickness Editor", "Bottom Thickness", self._update_airfoil_plot)

        # Create the airfoil plot window, passing references to the control points
        self.airfoil_plotter = AirfoilPlotWindow(
            self.root,
            self.camber_editor.control_points,
            self.top_thickness_editor.control_points,
            self.bottom_thickness_editor.control_points,
            self.camber_editor # Pass the camber editor instance
        )

        # Set geometry for each window
        # Top-left
        self.camber_editor.window.geometry(f"{window_width}x{window_height}+0+0")
        # Top-right
        self.top_thickness_editor.window.geometry(f"{window_width}x{window_height}+{window_width}+0")
        # Bottom-left
        self.bottom_thickness_editor.window.geometry(f"{window_width}x{window_height}+0+{window_height}")
        # Bottom-right
        self.airfoil_plotter.window.geometry(f"{window_width}x{window_height}+{window_width}+{window_height}")


        # Ensure windows close properly
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.camber_editor.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.top_thickness_editor.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.bottom_thickness_editor.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.airfoil_plotter.window.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Explicitly draw initial curves for each editor AFTER airfoil_plotter is initialized
        self.camber_editor.draw_curve()
        self.top_thickness_editor.draw_curve()
        self.bottom_thickness_editor.draw_curve()


    def _update_airfoil_plot(self):
        """
        Callback function invoked by editor windows to trigger a redraw
        of the main airfoil plot.
        """
        # Check if airfoil_plotter is initialized before calling its method
        if hasattr(self, 'airfoil_plotter') and self.airfoil_plotter is not None:
            self.airfoil_plotter.update_plot()

    def _on_closing(self):
        """Handles closing all windows when one is closed."""
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MainApplication()
    app.run()