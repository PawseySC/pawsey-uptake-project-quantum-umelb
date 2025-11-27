"""
@brief This runs a traf set to produce training/validation data for training a QML/ML

The traf run consists of several steps. The path of the executabes needs to be set

export PATH_EXE=/software/projects/bq2/mrosenzweig/uptake/executables/

Then you locall run several executables in sequence
exec_list = [
    "jerryo210310.x",
    "tomo.x",
    "addrad.x",
    "stitcho-q3d.x",
    "stitcho-q3d.x",
    "trafq3d22_14122022/trafq3d22.x"
]

a summary of a bash script is 

#create airfoil.dat using ??? and copy it to the working directory

$PATH_EXE/jerryo210310.x
$PATH_EXE/tomo.x < datatomo
$PATH_EXE/addrad.x < dataadd
cp -s meshq3d fort.12
$PATH_EXE/stitcho-q3d.x < datastitch_inlet
cp ./mq3d_stitch.dat fort.11
$PATH_EXE/stitcho-q3d.x < datastitch_outlet
cp ./mq3d_stitch.dat fort.13
cp -s xy.dat xy.xyz
$PATH_EXE/trafq3d22_14122022/trafq3d22.x 
cp -s xy.dat xy.xyz
cp -s fort.40 fort.50
cp -s fort.40 fort.70
$PATH_EXE/pstg2d14a_moverows.x < datapstg
cp -s fort.62 flow.dat
$PATH_EXE/outq3d12c.x
$PATH_EXE/pstg2d14a_moverows.x < datapstg
python3 $PATH_EXE/read_ptloss.py


"""

import sys, os, re

# import qbitbridge
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../qbitbridge/"
)
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../blade_design_tools/"
)
# import circuits
from time import sleep
import datetime
from typing import List, Set, Callable, Tuple, Dict, Any
from qbitbridge.vqpubase import HybridQuantumWorkflowBase
from qbitbridge.utils import upload_image_as_artifact
import asyncio
from prefect import task, flow
from prefect.logging import get_run_logger
import numpy as np
import subprocess
import shutil
import itertools
import hashlib
import pandas
import matplotlib.pyplot as plt
from pathlib import Path
import blade_design_tools.create_airfoil as ca


def copy_and_replace(
    source_file: str,
    destination_file: str,
    strings: List[Tuple[str, str]],
) -> bool:
    """
    Copies a file to a destination and replaces all occurrences of an old string with a new string in the copied file.

    Args:
        source_file (str): Path to the source file.
        destination_file (str): Path to where the copied file will be saved.
        strings (List[Tuple[str, str]]): List of old and new strings

    Returns:
        bool: True if the operation was successful, False otherwise.
    """

    try:
        # Copy the file
        shutil.copy2(source_file, destination_file)  # copy2 preserves metadata

        # Read the copied file
        with open(destination_file, "r") as f:
            content = f.read()
        for old_string, new_string in strings:
            # Replace the string
            content = content.replace(old_string, new_string)
        # Write the modified content back to the file
        with open(destination_file, "w") as f:
            f.write(content)
        return True  # Indicate success
    except Exception as e:
        print(f"An error occurred: {e}")  # Print the exception message for debugging
        return False  # Indicate failure


def update_traf_input(
    template_name: str,
    output_name: str,
    param_set: List[Tuple[str,str]],
) -> None:
    copy_and_replace(template_name, output_name, param_set)
    ca.generate_shape(config_filepath = output_name, make_plot = False)
    

def generate_shape(
        config_filepath="airfoil_config.yaml", 
        config_dict=None, 
        plot_foil=True
        ) -> None: #-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Docstring for generate_shape
    
    :param config_filepath: Description
    :param config_dict: Description
    :param plot_foil: Description
    """
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
            plot_name = config['output_settings']['output_airfoil_filename']+".png"
            plot_airfoil(
                fname=config['output_settings']['output_airfoil_filename'],
                oname=plot_name,
                # camber_curve_points, 
                # airfoil_top_points, 
                # airfoil_bottom_points,
                # config['output_settings']['plot_title']
            )
    
        ca.write_airfoil_dat(
            airfoil_top_points, 
            airfoil_bottom_points, 
            camber_curve_points,
            config['output_settings']['output_airfoil_filename'],
            config['output_settings']['chord_length_for_export']
        )


    except Exception as e:
        print(f"{e}")
        # might need to think how to handle this error in a flow. 
        pass

    #return airfoil_top_points, airfoil_bottom_points, camber_curve_points

def plot_airfoil(
        fname : str = "airfoil.dat", 
        oname : str = "airfoil.png"
         ) -> None:
    """plot the airfoil shape from the airfoil.dat file
    
    Args:
        fname (str, optional): airfoil file name. Defaults to "airfoil.dat".
        oname (str, optional): output image file name. Defaults to "airfoil.png".
    
    """
    offset = 0
    c_ax = np.loadtxt(fname,max_rows=1)
    offset += 1
    nss = np.loadtxt(fname,skiprows=offset,max_rows=1,dtype='int32')
    offset += 1
    ss = np.loadtxt(fname,skiprows=offset,max_rows=nss)
    offset += nss
    nps = np.loadtxt(fname,skiprows=offset,max_rows=1,dtype='int32')
    offset += 1
    ps = np.loadtxt(fname,skiprows=offset,max_rows=nps)
    blade = c_ax * np.concatenate((ss,np.flip(ps,axis=0)),axis=0)
    fig,ax = plt.subplots()
    ax.plot(blade[:,0],blade[:,1])
    ax.set_aspect('equal')
    fig.savefig(oname)
    asyncio.run(upload_image_as_artifact(Path(oname)))


def process_traf_output(
        fname : str = "pitch2",
) -> float:
    """process the traf output file to get the relevant output

    Args:
        fname (str, optional): traf output file name. Defaults to "pitch2".
    Returns:
        float: pressure loss
    """
    y,pt = np.genfromtxt(fname,skip_header=3,usecols=[0,5]).T
    ptloss = 1.0-np.trapz(pt,y)
    return ptloss

def process_and_write_traf_output(
        rawresults : Dict, 
        data : Dict,
        output_name : str) -> None:
    """process and save relevant output from all traf runs"""
    for key in rawresults.keys():
        i = data["pname"].index(key)
        data["TOTAL TO TOTAL ISENTROPIC EFFICIENCY"][i]=float(rawresults[key])
    df1 = pandas.DataFrame(data)
    # need to figure out best way of saving results
    # for now we produce a csv file with results 
    df1.to_csv(f"{output_name}.csv")

@task
def run_traf(
    param_set_name: str,
    param_set: List[Tuple[str, str]],
    airfoil_config_template: str,
    traf_script_template: str, 
    traf_ic_template_dir: str,
    outdir: str = "./",
) -> Any:
    """
    Task running the process to get multall output given a meangen input
    """
    if outdir[-1] != "/":
        outdir+="/"
    logger = get_run_logger()

    if os.path.isfile(f"{outdir}traf.log"):
        logger.info(f"Skipping traf as log file exists, just process pitch2")

        # parse traf output for relevant data 
        result = process_traf_output(f"{outdir}pitch2")
        return result

    logger.info(f"Running traf for {param_set_name}")
    os.makedirs(outdir, exist_ok=True)
    # copy the relevant traf input files to the output dir
    for item in os.listdir(traf_ic_template_dir):
        source_item_path = os.path.join(traf_ic_template_dir, item)
        destination_item_path = os.path.join(outdir, item)

        # Check if the item is a file
        if os.path.isfile(source_item_path):
            shutil.copy2(source_item_path, destination_item_path)
            logger.info(f"Copied: {source_item_path} to {destination_item_path}")

    info: Dict[str, str] = {}
    # create the airfoil updated config file
    update_traf_input(
        airfoil_config_template,
        f"{outdir}airfoil_config.yaml",
        param_set = param_set,
    )
    generate_shape(config_filepath=f"{outdir}airfoil_config.yaml")
    scriptname = f"{outdir}run_traf.sh"
    logger.info(f"Creating script {scriptname}")
    # copy script 
    shutil.copy2(traf_script_template, scriptname)

    logger.info(f"Running traf ...")
    process = subprocess.run(
        [scriptname],
        capture_output=True,
        text=True,
    )
    # do some post processing of output if required
    info = process.stdout
    with open(f"{outdir}traf.log", "w") as f:
        f.write(info)
    logger.info(f"Finished running traf")

    # parse traf output for relevant data 
    result = process_traf_output(f"{outdir}pitch2")
    return result

@task
def traf_create_param_set(params: Dict[str, Any]) -> Tuple[List, List]:
    """Create list of parameters with all possible combinations

    Args:
        params (Dict[str, Any]) : dictionary of list of parameter names and array of values

    Returns:
        Tuple of set names and param values
    """
    logger = get_run_logger()
    logger.info(f"Construct list of parameters and the relevant name of the set")
    param_values = list()
    param_names = list()
    for name, val in params.items():
        if val is not None:
            param_values.append(list(val))
            param_names.append(name)
    param_values = list(itertools.product(*param_values))
    return param_names, param_values


@flow(
    name="CPU flow running low traf simulations",
    flow_run_name="traf_cpu-{date:%Y-%m-%d:%H:%M:%S}",
    description="Running traf",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
async def traf_workflow(
    myqpuworkflow: HybridQuantumWorkflowBase,
    airfoil_config_template: str,
    traf_script_template: str,
    traf_ic_template_dir: str,
    params: Dict[str, Any],
    baseoutput_dir: str = "./",
    output_name : str = "traf_runs", 

    run_name : str = "run",
    hash_length : int = 6,
    resultkeys : List[str] = [
        "TOTAL TO TOTAL ISENTROPIC EFFICIENCY",
    ],
    max_task_submissions : int = 512,
    max_task_running : int = 1024,
    date: datetime.datetime = datetime.datetime.now(),
) -> None:
    """Flow for running traf 

    """
    logger = get_run_logger()
    logger.info(f"Launching traf CPU flow")

    param_names, param_values = traf_create_param_set.fn(params)
    logger.info(f"Number of parameter sets to run: {len(param_values)}")

    # initialize the data dictionary
    data = {
        "pname" : [None for i in range(len(param_values))],
        "outdir": [None for i in range(len(param_values))],
    }
    for key in param_names:
        data[key] = [None for i in range(len(param_values))]
    for key in resultkeys:
        data[key] = [None for i in range(len(param_values))]

    # submit the task and wait for results
    counter = 0
    active = 0 
    futures = {}
    results = {}
    for param_vals in param_values:
        pset = list()
        pname = list()
        for i in range(len(param_names)):
            pset.append((param_names[i], str(param_vals[i])))
            pname.append(f"{param_names[i]}_{param_vals[i]}")
        pname = "-".join(pname)
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()
        # Update the hash object with the input bytes
        sha256_hash.update(pname.encode('utf-8'))
        # take the first n from the hash
        hex_digest = sha256_hash.hexdigest()[:hash_length]
        outdir = f"{baseoutput_dir}/{run_name}-{hex_digest}/"
        pset.append(("AIRFOIL_FILENAME", f"{outdir}airfoil.dat"))
        
        if not os.path.isfile(f"{outdir}/traf.log"):
            futures[pname] = run_traf.submit(
                    param_set_name=pname,
                    airfoil_config_template=airfoil_config_template,
                    traf_script_template=traf_script_template,
                    traf_ic_template_dir=traf_ic_template_dir,
                    param_set=pset,
                    outdir=outdir,
                )
            active += 1
            
        else:
            futures[pname] = None
            results[pname] = run_traf.fn(
                    param_set_name=pname,
                    airfoil_config_template=airfoil_config_template,
                    traf_script_template=traf_script_template,
                    traf_ic_template_dir=traf_ic_template_dir,
                    param_set=pset,
                    outdir=outdir,
                )
            
        data["pname"][counter]=pname
        for p in param_names:
            data[p][counter]=param_vals[i]
        data["outdir"][counter]=outdir
        counter += 1
        if counter % max_task_submissions == 0:
            logger.info(f"Pausing task submission for 10 seconds to not overload database")
            sleep(10)
        while active >= max_task_running:
            logger.info(f"Pausing task submission till number active is reduced. ")
            sleep(100)
            for key in futures.keys():
                if futures[key] is not None:
                    if futures[key].state.is_final():
                        active -= 1

    # get the results from running the traf tasks
    for key in futures.keys():
        if futures[key] is not None:
            results[key] = futures[key].result()
    process_and_write_traf_output(results, data, f"{baseoutput_dir}/{output_name}")
    logger.info("Finished traf CPU flow")


def wrapper_to_async_flow(
    args: Any,
) -> None:
    """
    Run the traf runner
    """
    
    if args.params_file is not None:
        import json

        if not os.path.exists(args.params_file):
            raise ValueError(f"Parameter file {args.params_file} does not exist")
        with open(args.params_file, "r") as f:
            params = json.load(f)
    else:
        params: Dict[str, Any] = {
            # "TOP_INLET_ANGLE": np.arcsin(np.linspace(np.sin(-70./180.*np.pi), np.sin(70./180.*np.pi), num=5, endpoint=True))/np.pi*180.0,
            # "TOP_OUTLET_ANGLE": np.arcsin(np.linspace(np.sin(-70./180.*np.pi), np.sin(70./180.*np.pi), num=5, endpoint=True))/np.pi*180.0,
            # "BOTTOM_INLET_ANGLE": np.arcsin(np.linspace(np.sin(-70./180.*np.pi), np.sin(70./180.*np.pi), num=5, endpoint=True))/np.pi*180.0,
            # "BOTTOM_OUTLET_ANGLE": np.arcsin(np.linspace(np.sin(-70./180.*np.pi), np.sin(70./180.*np.pi), num=5, endpoint=True))/np.pi*180.0,
            "TOP_INLET_ANGLE": np.linspace(-40.0, 40.0, num=5, endpoint=True),
            "TOP_OUTLET_ANGLE": np.linspace(-40.0, 40.0, num=5, endpoint=True),
            "BOTTOM_INLET_ANGLE": np.linspace(-40.0, 40.0, num=5, endpoint=True),
            "BOTTOM_OUTLET_ANGLE": np.linspace(-40.0, 40.0, num=5, endpoint=True),
            "BLADE_SHAPE": None,
        }

    myflow = HybridQuantumWorkflowBase(
        cluster=args.cluster,
        vqpu_ids=[1, 2, 3, 16],
        vqpu_template_yaml=args.yaml_template,
        vqpu_template_script=args.script_template,
        eventloc=f"{os.path.dirname(os.path.abspath(__file__))}/events/",
    )

    if args.traf_script_template is not None:
        if not os.path.exists(args.traf_script_template):
            raise ValueError(f"traf file {args.traf_script_template} does not exist")
    if args.airfoil_config_template is not None:
        if not os.path.exists(args.airfoil_config_template):
            raise ValueError(f"airfoil config file {args.airfoil_config_template} does not exist")

    asyncio.run(
        traf_workflow.with_options(task_runner=myflow.gettaskrunner("cpu-single"))(
            myqpuworkflow = myflow,
            airfoil_config_template = args.airfoil_config_template,
            traf_script_template = args.traf_script_template,
            traf_ic_template_dir = args.traf_ic_template_dir,
            params = params,
            baseoutput_dir = args.output_dir,
            run_name = args.run_name,
            output_name = args.output_name,
        )
    )

def traf_argparse():
    import argparse
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Traf workflow arguments.")

    # Define arguments
    parser.add_argument("--output_dir", type = str, default = "./", help="output path for results")
    parser.add_argument("--yaml_template", type=str, 
                        default=f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/remote_vqpu_template.example.yaml", 
                        help="yaml file for the vqpu")
    parser.add_argument("--script_template", type=str, 
                        default=f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/vqpu_template.example.sh",
                        help="script template file for the vqpu")
    parser.add_argument("--cluster", type=str, default="traf-setonix-pypath", help="cluster name for running workflow")
    parser.add_argument("--airfoil_config_template", type=str, default="./airfoil_config.template.yaml", help="template for airfoil config input file")
    parser.add_argument("--traf_script_template", type=str, default="./run_traf.template.sh", help="template for traf run script")
    parser.add_argument("--traf_ic_template_dir", type=str, default="/software/projects/bq2/mrosenzweig/uptake/test_dummy/", help="base dir containing ic files relevant traf simulation files to copy")
    parser.add_argument("--run_name", type=str, default="run", help="name for the run")
    parser.add_argument("--output_name", type=str, default="traf_runs", help="name for the output file")
    parser.add_argument("--params_file", type=str, default=None, help="json file containing the parameters to run")

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = traf_argparse()
    wrapper_to_async_flow(args=args)
