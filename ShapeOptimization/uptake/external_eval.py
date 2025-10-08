import numpy as np
import sys
import os
import copy
import contextlib

import multiprocessing as mp
from multiprocessing import Pool
import subprocess
import matplotlib.pyplot as plt

import json
print("Ext. wrapper called")
os.chdir("uptake")
sys.path.append(os.path.abspath("../../blade_design_tools"))
import create_airfoil as ca


#modified shape_method - need controled file write!
def generate_shape(config_filepath="airfoil_config.yaml", config_dict=None, plot_foil=False):
    try:
        config = ca.load_config(config_filepath) if config_dict is None else config_dict

        # Handle optional input targets for chord and max thickness
        input_targets = config.get('input_targets', {})
        desired_chord = input_targets.get('desired_chord_length', None)
        desired_max_th = input_targets.get('desired_max_thickness', None)

        # Temporarily compute camber to get normalized chord (needed for adjustments)
        temp_camber_control_points, temp_camber_curve_points = ca.compute_curve_points(
            config['camber_line'], "camber", config['output_settings']['num_points_on_curve']
        )
        norm_le = temp_camber_curve_points[0]
        norm_te = temp_camber_curve_points[-1]
        norm_chord = np.linalg.norm(norm_te - norm_le)
        norm_x_span = norm_te[0] - norm_le[0]  # Should be 1.0

        # Adjust chord_length_for_export if desired_chord is set (to achieve desired geometric chord)
        if desired_chord is not None:
            config['output_settings']['chord_length_for_export'] = desired_chord * (norm_x_span / norm_chord)  # Equivalent to desired_chord * np.cos(np.deg2rad(np.abs(config['camber_line']['stagger_angle_deg'])))

        # 1. Compute Camber Line
        camber_control_points, camber_curve_points = ca.compute_curve_points(
            config['camber_line'], "camber", config['output_settings']['num_points_on_curve']
        )

        # 2. Compute Top Thickness Distribution
        top_thickness_control_points, top_thickness_curve_points = ca.compute_curve_points(
            config['top_thickness'], "top_thickness", config['output_settings']['num_points_on_curve']
        )
        # 3. Compute Bottom Thickness Distribution
        bottom_thickness_control_points, bottom_thickness_curve_points = ca.compute_curve_points(
            config['bottom_thickness'], "bottom_thickness", config['output_settings']['num_points_on_curve']
        )

        # Temporarily compute current normalized max thickness
        current_norm_thickness_along = top_thickness_curve_points[:, 1] + bottom_thickness_curve_points[:, 1]
        current_norm_max_th = np.max(current_norm_thickness_along)

        # Compute scale factor (reflects any chord adjustment)
        scale = config['output_settings']['chord_length_for_export'] / norm_x_span

        # Adjust thickness curves if desired_max_th is set
        if desired_max_th is not None:
            desired_norm_max_th = desired_max_th / scale
            th_scale_factor = desired_norm_max_th / current_norm_max_th
            top_thickness_curve_points[:, 1] *= th_scale_factor
            bottom_thickness_curve_points[:, 1] *= th_scale_factor

        # 4. Generate Full Airfoil Contour
        airfoil_top_points, airfoil_bottom_points = ca.generate_airfoil_contour(
            camber_curve_points,
            top_thickness_curve_points,
            bottom_thickness_curve_points,
            config['camber_line']['inlet_angle_deg'],
            config['output_settings']['num_points_on_arc']
        )
        # 5. Plot the Airfoil
        if plot_foil:
            ca.plot_airfoil(camber_curve_points, airfoil_top_points, airfoil_bottom_points,
                         config['output_settings']['plot_title'])

        if config['output_settings']['output_airfoil_filename'] != "None":
            ca.write_airfoil_dat(airfoil_top_points, airfoil_bottom_points, camber_curve_points,
                            config['output_settings']['output_airfoil_filename'],
                            config['output_settings']['chord_length_for_export'])


    except Exception as e:
        print(f"{e}")
        pass

    return airfoil_top_points, airfoil_bottom_points, camber_curve_points


config_ = ca.load_config("../../blade_design_tools/airfoil_config.yaml")

action_idx = {
    1: ["camber_line","control_points_params","P1_k_factor",(0.15,0.5)],
    2: ["camber_line","control_points_params","P3_k_factor",(-0.3,-0.1)],
    3: ["top_thickness","control_points_params","P1_k_factor",(0.15,0.4)],
    4: ["top_thickness","le_y_thickness",(0.01,0.04)]
}

control_params = {
    "top_thickness" : "bottom_thickness"
}

qoi = "pitch2"

def run_script(script_path):
    result = subprocess.run(
        [script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True  # only needed if script is not executable
    )
    return result.stdout, result.stderr, result.returncode

def fail_ret_no(def_path, new_f):
    os.chdir(def_path)
    os.system(f"rm -rf {new_f}")
    return 9999

def external_eval_no(actions_, run_id, plot_foil=False):  # action trajectoroy consist of max 4
    actions = actions_
    new_f = f"test{run_id}"
    cwd = os.getcwd()
    os.system(f"cp -r dummy {new_f}")
    os.chdir(new_f)

    config = copy.copy(config_)
    for step, action_ in enumerate(actions):
        action_data = action_idx[step+1]
        lb = action_data[-1][0]
        ub = action_data[-1][1]

        if len(action_data) == 4:
            config[action_data[0]][action_data[1]][action_data[2]] = lb + action_ * (ub- lb)
    
    
        #for simplicity just consider one thick. param
        if len(action_data) == 3:
            config[action_data[0]][action_data[1]] = lb + action_ * (ub - lb)
            config[control_params[action_data[0]]][action_data[1]] = lb + action_ * (ub - lb)

    config['output_settings']['output_airfoil_filename'] = f"airfoil.dat"
    airfoil_top_points, airfoil_bottom_points, _ = generate_shape(config_dict=config, plot_foil=plot_foil)

    #cp test dummy stuff
    
    
    
    #os.system(f"mv airfoil{run_id}.dat {new_f}/airfoil.dat")
    
    print(f"Now folder = {os.getcwd()}")
    
    os.system("../scripts/build.sh")
    os.system("../scripts/run_traf.sh")
    os.system("../scripts/do_post.sh")

    #calc loss
    try:
        y,pt = np.genfromtxt(qoi,skip_header=3,usecols=[0,5]).T
        loss = 1-np.trapz(pt,y)
    except:
        return fail_ret_no(cwd, new_f)

    os.chdir(cwd)
    print(f"Run_id: {run_id}")

    np.savetxt(f"{new_f}/loss.dat",[loss])


if __name__ == '__main__':
    actions_str = sys.argv[1]
    run_id = sys.argv[2]
    plot_foil = sys.argv[3].lower() == 'true'
    
    actions =json.loads(actions_str)  # Deserialize list
    #with open(os.devnull, 'w') as devnull:
    #with contextlib.redirect_stdout(devnull):
    external_eval_no(actions, run_id, True)
