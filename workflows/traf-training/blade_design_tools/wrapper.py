import numpy as np
import matplotlib.pyplot as plt

from create_airfoil import *

def main():
    config_filepath = "airfoil_config.yaml"
    config = load_config(config_filepath)
    
    fig,ax = plt.subplots()
    for fact in [0.02,0.03,0.04]:
        config["top_thickness"]["control_points_params"]["P2_y"] = fact
        config["bottom_thickness"]["control_points_params"]["P2_y"] = fact

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
        xx = np.concatenate((airfoil_top_points[:,0],np.flip(airfoil_bottom_points[:,0],axis=-1)),axis=0)
        yy = np.concatenate((airfoil_top_points[:,1],np.flip(airfoil_bottom_points[:,1],axis=-1)),axis=0)
        ax.plot(xx,yy,label=fact)

    ax.set_aspect("equal")
    ax.legend()
    plt.show()
if __name__ == "__main__":
    main()
