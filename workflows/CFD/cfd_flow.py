"""
@brief This example shows how to structure a multi-vqpu workflow.

This workflow spins up two or more vqpus and then has a workflow that runs cpu/gpu flows that also then spawn circuit flows.

"""

import sys, os, re

# import qbitbridge
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../qbitbridge/")
# import circuits
from time import sleep
import datetime
from typing import List, Set, Callable, Tuple, Dict
from qbitbridge.options import vQPUWorkflow
from qbitbridge.vqpubase import HybridQuantumWorkflowBase
from qbitbridge.vqpuflow import (
    launch_vqpu_workflow,
    circuits_with_nqvpuqs_workflow,
    circuits_vqpu_workflow,
    run_cpu,
    run_gpu, 
)
from qbitbridge.vqpufitting import (
    LikelihoodModel,
    LikelihoodModelRuntime,
    model_fit_and_analyse_workflow,
    multi_model_flow,
)
from qbitbridge.utils import EventFile, save_artifact
from workflow.circuits.qristal_circuits import simulator_setup, noisy_circuit
import asyncio
from prefect import flow
from prefect_dask import DaskTaskRunner
from prefect.logging import get_run_logger
import numpy as np


@flow(
    name="Example CPU flow running low res CFD simulations ",
    flow_run_name="cfd_cpu-{date:%Y-%m-%d:%H:%M:%S}",
    description="Running Hipstar CFD low res simulations",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
async def cfd_cpu_workflow(
    myqpuworkflow: HybridQuantumWorkflowBase,
    cfdexec : str, 
    cfdargs : List[Any], 
    date: datetime.datetime = datetime.datetime.now(),
) -> None:
    """Flow for running cpu based programs.

    arguments (str): string of arguments to pass extra options to run cpu
    """
    logger = get_run_logger()
    logger.info("Launching CFD CPU flow")
    # submit the task and wait for results
    futures = []
    for args in cfdargs:
        logger.info(f"Running {cfdexec} with {args}")
        futures.append(
            await run_cpu.submit(myqpuworkflow=myqpuworkflow, exec=exec, arguments=args)
        )
    for f in futures:
        await f.result()

    logger.info("Finished CFD CPU flow")

@flow(
    name="Example GPU flow running high res CFD simulations ",
    flow_run_name="cfd_cpu-{date:%Y-%m-%d:%H:%M:%S}",
    description="Running Hipstar CFD high res simulations",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
async def cfd_gpu_workflow(
    myqpuworkflow: HybridQuantumWorkflowBase,
    cfdexec : str, 
    cfdargs : List[Any], 
    date: datetime.datetime = datetime.datetime.now(),
) -> None:
    """Flow for running gpu based programs.

    arguments (str): string of arguments to pass extra options to run cpu
    """
    logger = get_run_logger()
    logger.info("Launching CFD GPU flow")
    # submit the task and wait for results
    futures = []
    for args in cfdargs:
        logger.info(f"Running {cfdexec} with {args}")
        futures.append(
            await run_gpu.submit(myqpuworkflow=myqpuworkflow, exec=exec, arguments=args)
        )
    for f in futures:
        await f.result()

    logger.info("Finished CFD GPU flow")


@flow(
    name="Multi-CFD sim run",
    flow_run_name="cfd-{date:%Y-%m-%d:%H:%M:%S}",
    description="Running cfd sims ",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
async def cfd_workflow(
    myqpuworkflow: HybridQuantumWorkflowBase,
    cpuexecs: str,
    cpuargs: List[str],
    gpuexecs: str,
    gpuargs: List[str],
    date: datetime.datetime = datetime.datetime.now(),
):
    """
    CFD sims
    """

    logger = get_run_logger()
    logger.info("Running CFD sims workflow")

    cpuflows = cfd_cpu_workflow.with_options(
        task_runner=myqpuworkflow.gettaskrunner("cpu"),
    )
    gpuflows = cfd_gpu_workflow.with_options(
        task_runner=myqpuworkflow.gettaskrunner("multigpu")
    )

    async with asyncio.TaskGroup() as tg:
        # either spin up real vqpu
        tg.create_task(
            cpuflows(
                myqpuworkflow=myqpuworkflow,
                cfdexec=cpuexecs,
                cfdargs=cpuargs,
            )
        )

        tg.create_task(
            gpuflows(
                myqpuworkflow=myqpuworkflow,
                cfdexec=gpuexecs,
                cfdargs=gpuargs,
            )
        )

    logger.info("Finished CFD workflow")

def wrapper_to_async_flow(
    yaml_template: str | None = None,
    script_template: str | None = None,
    cluster: str | None = None,
    cpuexecs: str = "hipstar-cpu",
    cpuargs: List[str] = [
        "something",
        "somethingelse",
    ],
    gpuexecs: str = "hipstar-gpu",
    gpuargs: List[str] = [
        "something", 
        "somethingelse",
        ],
) -> None:
    """
    Run the CFD runner 
    """
    if yaml_template == None:
        yaml_template = f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/remote_vqpu_ella_template.yaml"
    if script_template == None:
        script_template = f"{os.path.dirname(os.path.abspath(__file__))}/../../qbitbridge/workflow/qb-vqpu/vqpu_template_ella_qpu-1.7.0.sh"
    if cluster == None:
        cluster = "cfd_setonix_flow"
    myflow = HybridQuantumWorkflowBase(
        cluster=cluster,
        vqpu_ids=[1, 2, 3, 16],
        vqpu_template_yaml=yaml_template,
        vqpu_template_script=script_template,
        eventloc=f"{os.path.dirname(os.path.abspath(__file__))}/events/",
    )

    asyncio.run(
        cfd_workflow(
            myqpuworkflow=myflow,
            cpuexecs=cpuexecs,
            cpuargs=cpuargs,
            gpuexecs=gpuexecs,
            gpuargs=gpuargs,
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
