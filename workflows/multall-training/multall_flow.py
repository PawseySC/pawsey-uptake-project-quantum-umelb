"""
@brief This runs a multall set to produce training/validation data for training a QML/ML

"""

import sys, os, re

# import qbitbridge
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../qbitbridge/"
)
# import circuits
from time import sleep
import datetime
from typing import List, Set, Callable, Tuple, Dict, Any
from qbitbridge.vqpubase import HybridQuantumWorkflowBase
import asyncio
from prefect import task, flow, concurrency
from prefect.logging import get_run_logger
import numpy as np
import subprocess
import shutil
import itertools
import hashlib
import pandas

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


def update_multall_input(
    template_name: str,
    output_name: str,
    param_set: List[Tuple[str,str]],
) -> None:
    copy_and_replace(template_name, output_name, param_set)
    

def process_multall_output(
        input : str, 
        inletkeystrings : List[str] = [
            "INLET AND EXIT STAGNATION PRESSURES",
            "INLET AND EXIT STATIC PRESSURES",
            "INLET AND EXIT STAGNATION TEMPERATURES",
            "INLET AND OUTLET MASS FLOW RATES",
        ],
        efficiencykeystrings : List[str] = [
            "TOTAL TO TOTAL ISENTROPIC EFFICIENCY",
            "TOTAL TO STATIC ISENTROPIC EFFICIENCY",
            "TOTAL TO TOTAL POLYTROPIC EFFICIENCY",
        ],
        tempkeystring: List[str]= [
            "INLET, MID and EXIT STATIC TEMPERATURES",
        ]
) -> Dict:
    # convert the string to a list find last instance of relevant strings
    input = input.strip().split('\n')
    input.reverse()
    data = {}
    for key in inletkeystrings:
        for i in input:
            if key in i:
                data[key] = str(i.split("=")[-1]).split()
                break
        
    for key in efficiencykeystrings:
        for i in input:
            if key in i:
                data[key] = (i.split("=")[-1]).split()
                break

    for key in tempkeystring:
        for i in input:
            if key in i:
                data[key] = str(i.split("=")[-1]).split()
                break
    return data 

def process_and_write_multall_output(
        rawresults : Dict, 
        data : Dict,
        output_name : str) -> None:
    """process and save relevant output from all multall runs"""
    for key in rawresults.keys():
        i = data["pname"].index(key)
        data["TOTAL TO TOTAL ISENTROPIC EFFICIENCY"][i]=float(rawresults[key]["TOTAL TO TOTAL ISENTROPIC EFFICIENCY"][0])
        data["EXIT STAGNATION TEMPERATURE"][i]=float(rawresults[key]["INLET AND EXIT STAGNATION TEMPERATURES"][1])
        data["EXIT STAGNATION PRESSURE"][i]=float(rawresults[key]["INLET AND EXIT STAGNATION PRESSURES"][1])
    df1 = pandas.DataFrame(data)
    # need to figure out best way of saving results
    # for now we produce a csv file with results 
    df1.to_csv(f"{output_name}.csv")


@task
def run_multall(
    param_set_name: str,
    param_set: List[Tuple[str, str]],
    multallexecs: List[str],
    execdir : str ,
    meangen_template: str | None = None,
    stagen_template: str | None = None,
    outdir: str = "./",
) -> Any:
    """
    Task running the process to get multall output given a meangen input
    """
    # with concurrency("data-processing", occupy=256):
    if outdir[-1] != "/":
        outdir+="/"
    logger = get_run_logger()
    logger.info(f"Running multall for {param_set_name}")
    os.makedirs(outdir, exist_ok=True)

    info: Dict[str, str] = {}
    if meangen_template is not None:
        update_multall_input(
            meangen_template,
            f"{outdir}meangen_{param_set_name}.in",
            param_set = param_set,
        )
    if stagen_template is not None:
        update_multall_input(
            stagen_template,
            f"{outdir}meangen_{param_set_name}_stagen.dat",
            param_set = param_set,
        )
        multallexecs = ["stagen", "multall"]

    for cmd in multallexecs:
        if os.path.isfile(f"{outdir}{cmd}.log"):
            logger.info(f"Skipping {cmd} as log file exists")
            with open(f"{outdir}{cmd}.log", "r") as f:
                info[cmd] = "".join(f.readlines())
        else:
            scriptname = f"{outdir}run_{cmd}.sh"
            with open(scriptname, 'w') as f:
                f.write(f"#!/bin/bash\n")
                f.write(f"cd {outdir} \n")
                f.write(f"export MEANGEN_ARGS=\"meangen_{param_set_name}.in meangen_{param_set_name} \"\n")
                f.write(f"export STAGEN_ARGS=\"meangen_{param_set_name}_stagen.dat stagen_out_{param_set_name} \"\n")
                f.write(f"export MULTALL_ARGS=\"stagen_out_{param_set_name}_new.dat multall_{param_set_name} \"\n")
                f.write(f"{execdir}/{cmd}\n")
            os.chmod(scriptname, 0o755)
            logger.info(f"Running {cmd} ...")
            process = subprocess.run(
                [scriptname],
                capture_output=True,
                text=True,
            )        
            # do some post processing of output if required
            info[cmd] = process.stdout
            with open(f"{outdir}{cmd}.log", "w") as f:
                f.write(info[cmd])
        logger.info(f"Finished running {cmd}")

    # parse the info from multall
    result = process_multall_output(info["multall"])
    return result


@task
def multall_create_param_set(params: Dict[str, Any]) -> Tuple[List, List]:
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
    name="CPU flow running low multall simulations",
    flow_run_name="multall_cpu-{date:%Y-%m-%d:%H:%M:%S}",
    description="Running multall",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
async def multall_workflow(
    myqpuworkflow: HybridQuantumWorkflowBase,
    meangen_template: str | None,
    stagen_template: str | None, 
    params: Dict[str, Any],
    baseoutput_dir: str = "./",
    output_name : str = "multall_runs", 
    run_name : str = "run",
    multallexecs: List[str] = ["meangen", "stagen", "multall"],
    execdir : str = "/software/projects/pawsey0001/pelahi/pawsey-uptake-project-quantum-umelb/CFD/multall/bin/",
    hash_length : int = 10,
    resultkeys : List[str] = [
        "INLET STAGNATION PRESSURE",
        "EXIT STAGNATION PRESSURE",
        "INLET STATIC PRESSURE",
        "EXIT STATIC PRESSURE",
        "INLET STAGNATION TEMPERATURE",
        "EXIT STAGNATION TEMPERATURE",
        "INLET MASS OUTFLOW RATE",
        "EXIT MASS OUTFLOW RATE",
        "TOTAL TO TOTAL ISENTROPIC EFFICIENCY",
        "TOTAL TO STATIC ISENTROPIC EFFICIENCY",
        "TOTAL TO TOTAL POLYTROPIC EFFICIENCY",
        "INLET STATIC TEMPERATURE",
        "MID STATIC TEMPERATURES",
        "EXIT STATIC TEMPERATURES",
    ],
    max_task_submissions : int = 256,
    date: datetime.datetime = datetime.datetime.now(),
) -> None:
    """Flow for running multall 

    """
    logger = get_run_logger()
    logger.info(f"Launching multall CPU flow")
    for exec in multallexecs:
        if not os.path.exists(f"{execdir}/{exec}"):
            raise ValueError(f"Executable {exec} in path {execdir} does not exist")

    param_names, param_values = multall_create_param_set.fn(params)

    # initialize the data dictionary
    data = {
        "pname" : [None for i in range(len(param_values))],
        "outdir": [None for i in range(len(param_values))],
    }
    for key in param_names:
        data[key] = [None for i in range(len(param_values))]
    for key in resultkeys:
        data[key] = [None for i in range(len(param_values))]
    logger.info(f"{data.keys()}")

    # submit the task and wait for results
    counter = 0
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
        if not os.path.isfile(f"{outdir}/multall.log"):
            futures[pname] = run_multall.submit(
                    param_set_name=pname,
                    param_set=pset,
                    multallexecs=multallexecs,
                    execdir=execdir,
                    meangen_template=meangen_template,
                    stagen_template=stagen_template,
                    outdir=outdir,

                )
        else:
            futures[pname] = None
            results[pname] = run_multall.fn(
                    param_set_name=pname,
                    param_set=pset,
                    multallexecs=multallexecs,
                    execdir=execdir,
                    meangen_template=meangen_template,
                    stagen_template=stagen_template,
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
    # get the results from running the multall tasks
    for key in futures.keys():
        if futures[key] is not None:
            results[key] = futures[key].result()
    logger.info(f"{results}")
    # logger.info(results[key])
    process_and_write_multall_output(results, data, f"{baseoutput_dir}/{output_name}")
    logger.info("Finished multall CPU flow")


def wrapper_to_async_flow(
    args: Any,
) -> None:
    """
    Run the multall runner
    """
    
    if args.params_file is not None:
        import json

        if not os.path.exists(args.params_file):
            raise ValueError(f"Parameter file {args.params_file} does not exist")
        with open(args.params_file, "r") as f:
            params = json.load(f)
    else:
        params: Dict[str, Any] = {
            "FLOW_ANGLE": np.arcsin(np.linspace(np.sin(-70/180*np.pi), np.sin(70/180*np.pi), num=255, endpoint=True))/np.pi*180,
            "FLOW_SPEED": None,
            "BLADE_SHAPE": None,
        }

    myflow = HybridQuantumWorkflowBase(
        cluster=args.cluster,
        vqpu_ids=[1, 2, 3, 16],
        vqpu_template_yaml=args.yaml_template,
        vqpu_template_script=args.script_template,
        eventloc=f"{os.path.dirname(os.path.abspath(__file__))}/events/",
    )

    if args.stagen_template is not None:
        if not os.path.exists(args.stagen_template):
            raise ValueError(f"Stagen file {args.stagen_template} does not exist")
        # just bypass meangen if stagen template is used 
        args.meangen_template = None

    if args.meangen_template is not None:
        if not os.path.exists(args.meangen_template):
            raise ValueError(f"Meangen file {args.meangen_template} does not exist")

    asyncio.run(
        multall_workflow.with_options(task_runner=myflow.gettaskrunner("cpu-single"))(
            myqpuworkflow = myflow,
            meangen_template = args.meangen_template,
            stagen_template = args.stagen_template, 
            params = params,
            baseoutput_dir = args.output_dir,
            run_name = args.run_name,
            output_name = args.output_name,
        )
    )

def multall_argparse():
    import argparse
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Multall workflow arguments.")

    # Define arguments
    parser.add_argument("--output_dir", type = str, default = "./", help="output path for results")
    parser.add_argument("--yaml_template", type=str, 
                        default=f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/remote_vqpu_template.example.yaml", 
                        help="yaml file for the vqpu")
    parser.add_argument("--script_template", type=str, 
                        default=f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/vqpu_template.example.sh",
                        help="script template file for the vqpu")
    parser.add_argument("--cluster", type=str, default="multall-setonix-pypath", help="cluster name for running workflow")
    parser.add_argument("--meangen_template", type=str, default="./meangen.in.template", help="template for meangen input file")
    parser.add_argument("--stagen_template", type=str, default=None, help="template for stagen input file")
    parser.add_argument("--run_name", type=str, default="run", help="name for the run")
    parser.add_argument("--output_name", type=str, default="multall_runs", help="name for the output file")
    parser.add_argument("--params_file", type=str, default=None, help="json file containing the parameters to run")

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = multall_argparse()
    wrapper_to_async_flow(args=args)
