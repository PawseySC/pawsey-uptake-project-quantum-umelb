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


def update_meangen_in(
    template_name: str,
    output_name: str,
    param_set: List[Tuple[str,str]],
) -> None:
    copy_and_replace(template_name, output_name, param_set)
    pass


@task
def run_multall(
    param_set_name: str,
    meangen_template: str,
    param_set: List[Tuple[str, str]],
    outdir: str = "./",
    multallexecs: List[str] = ["meangen", "stagen", "multall"],
    execdir : str = "/software/projects/pawsey0001/pelahi/pawsey-uptake-project-quantum-umelb/CFD/multall/bin/"
) -> Any:
    """
    Task running the process to get multall output given a meangen input
    """
    if outdir[-1] != "/":
        outdir+="/"
    os.environ["MEANGEN_ARGS"] = (
        f"{outdir}meangen_{param_set_name}.in {outdir}meangen_{param_set_name} "
    )
    os.environ["STAGEN_ARGS"] = (
        f"{outdir}/meangen_{param_set_name}_stagen.dat {outdir}stagen_out_{param_set_name} "
    )
    os.environ["MULTALL_ARGS"] = (
        f"{outdir}/stagen_out_{param_set_name}_new.dat {outdir}multall_{param_set_name} "
    )
    logger = get_run_logger()
    logger.info(f"Running multall for {param_set_name}")
    os.mkdir(outdir)

    info: Dict[str, str] = {}
    update_meangen_in(
        meangen_template,
        f"{outdir}meangen_{param_set_name}.in",
        param_set = param_set,
    )
    for cmd in multallexecs:
        process = subprocess.run(
            [f"{execdir}/{cmd}"],
            capture_output=True,
            text=True,
        )
        # do some post processing of output if required
        info[cmd] = process.stdout
        logger.info(f"{cmd}")
        logger.info(f"{info[cmd]}")
    # parse the info from multall
    result = info["multall"]

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
    meangen_template: str,
    params: Dict[str, Any],
    run_name : str = "run",
    date: datetime.datetime = datetime.datetime.now(),
) -> None:
    """Flow for running cpu based programs.

    arguments (str): string of arguments to pass extra options to run cpu
    """
    logger = get_run_logger()
    logger.info(f"Launching multall CPU flow")
    param_names, param_values = multall_create_param_set.fn(params)

    # submit the task and wait for results
    futures = []
    counter = 1
    for param_vals in param_values:
        pset = list()
        pname = ""
        for i in range(len(param_names)):
            pset.append((param_names[i], str(param_vals[i])))
            pname += f"{param_names[i]}={param_vals[i]}-"
        pname = pname[:-1]
        pname = pname.encode('utf-8')
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()
        # Update the hash object with the input bytes
        sha256_hash.update(pname)   
        hex_digest = sha256_hash.hexdigest()
        outdir = f"{run_name}-{hex_digest}/"
        if not os.path.isdir(outdir):
            futures.append(
                run_multall.submit(
                    param_set_name=pname,
                    param_set=pset,
                    meangen_template=meangen_template,
                    outdir=outdir,
                )
            )
        counter += 1
    for f in futures:
        f.result()

    logger.info("Finished multall CPU flow")


def wrapper_to_async_flow(
    meangen_template: str = "./meangen.in.template",
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

    # construct

    asyncio.run(
        multall_workflow.with_options(task_runner=myflow.gettaskrunner("cpu"))(
            myqpuworkflow=myflow,
            meangen_template=meangen_template, 
            params = params
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
