"""
@brief This runs a multallt set to produce training/validation data for training a QML/ML

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
from prefect import task, flow
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
    for key in results.keys():
        i = data["pname"].index(key)
        data["TOTAL TO TOTAL ISENTROPIC EFFICIENCY"][i]=float(results[key]["TOTAL TO TOTAL ISENTROPIC EFFICIENCY"][0])
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
    if outdir[-1] != "/":
        outdir+="/"
    logger = get_run_logger()
    logger.info(f"Running multall for {param_set_name}")
    os.mkdir(outdir)

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
    output_name : str = "multall_runs", 
    run_name : str = "run",
    multallexecs: List[str] = ["meangen", "stagen", "multall"],
    execdir : str = "/software/projects/pawsey0001/pelahi/pawsey-uptake-project-quantum-umelb/CFD/multall/bin/",
    hash_length : int = 6,
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

    # submit the task and wait for results
    counter = 0
    futures = {}
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
        outdir = f"{run_name}-{hex_digest}/"
        if not os.path.isdir(outdir):
            futures[pname] = run_multall.submit(
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
    results = {}
    # get the results from running the multall tasks
    for key in futures.keys():
        results[key] = futures[key].result()
    process_and_write_multall_output(results, data, output_name)
    logger.info("Finished multall CPU flow")


def wrapper_to_async_flow(
    meangen_template: str = "./meangen.in.template",
    stagen_template: str | None = None, #"./stagen.in.template",
    params: Dict[str, Any] = {
        "FLOW_ANGLE": np.arange(0, 10, 5),
        "FLOW_SPEED": None,
        "BLADE_SHAPE": None,
    },
    yaml_template: str | None = None,
    script_template: str | None = None,
    cluster: str | None = None,
) -> None:
    """
    Run the multall runner
    """
    if yaml_template == None:
        yaml_template = f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/remote_vqpu_template.example.yaml"
    if script_template == None:
        script_template = f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/vqpu_template.example.sh"
    if cluster == None:
        cluster = "multall-setonix-pypath"
    myflow = HybridQuantumWorkflowBase(
        cluster=cluster,
        vqpu_ids=[1, 2, 3, 16],
        vqpu_template_yaml=yaml_template,
        vqpu_template_script=script_template,
        eventloc=f"{os.path.dirname(os.path.abspath(__file__))}/events/",
    )

    if stagen_template is not None:
        if not os.path.exists(stagen_template):
            raise ValueError(f"Stagen file {stagen_template} does not exist")
        # just bypass meangen if stagen template is used 
        meangen_template = None

    if meangen_template is not None:
        if not os.path.exists(meangen_template):
            raise ValueError(f"Meangen file {meangen_template} does not exist")

    asyncio.run(
        multall_workflow.with_options(task_runner=myflow.gettaskrunner("cpu"))(
            myqpuworkflow=myflow,
            meangen_template=meangen_template,
            stagen_template=stagen_template, 
            params = params,
        )
    )

if __name__ == "__main__":
    yaml_template = None
    script_template = None
    cluster = None
    res = [i for i in sys.argv if re.findall("--yaml=", i)]
    if len(res) > 0:
        yaml_template = res[0].split("=")[1]
    res = [i for i in sys.argv if re.findall("--script=", i)]
    if len(res) > 0:
        script_template = res[0].split("=")[1]
    res = [i for i in sys.argv if re.findall("--cluster=", i)]
    if len(res) > 0:
        cluster = res[0].split("=")[1]
    wrapper_to_async_flow(
        yaml_template=yaml_template,
        script_template=script_template,
        cluster=cluster,
    )
