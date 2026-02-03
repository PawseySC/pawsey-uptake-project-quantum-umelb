# coding: utf-8

# In[1]:


# general imports
import os
os.environ["OMP_NUM_THREADS"] = "8"

import time
import argparse
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import pandas as pd


# CLI

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, metavar="#qubits")
parser.add_argument("-l", type=int, metavar="#layers")
parser.add_argument("-r", type=int, metavar="#rotations")
parser.add_argument("-e", action="store_true") # reencode
parser.add_argument("-d", choices=["cpu", "gpu"], default="gpu", metavar="device")
parser.add_argument("-m", type=int, default=100, metavar="maxiter")
parser.add_argument("-t", type=float, default=1e-4, metavar="tolerance")
parser.add_argument("-g", type=float, default=1e-3, metavar="gradient tolerance")
parser.add_argument("-i", default="traf_runs.csv")
parser.add_argument("-o", default=".")
parser.add_argument("-v", action="store_true")

args = parser.parse_args()
print("Arguments", args)

infile = args.i
outdir = args.o
verbose = args.v


# Circuit parameters
n_qubits = args.q
layers = args.l
global_rots = args.r
reencode = args.e
enc_layers = [0]

if args.d == "cpu":
    device = qml.device("lightning.qubit", wires=n_qubits)
elif args.d == "gpu":
    device = qml.device("lightning.gpu", wires=n_qubits)
else:
    raise Exception(f"Unknown device, got {args.d}")

options = {"maxiter": args.m, "term": args.t, "grad_term": args.g}


# Circuit design
if global_rots not in [2, 3]:
    raise Exception(f"global rotations must be 2 or 3, got {global_rots}")

def rotations(wire, params):
    qml.RZ(params[0], wires=wire)
    qml.RY(params[1], wires=wire)
    if global_rots == 3:
        qml.RZ(params[2], wires=wire)


def entangle(n_qubits):
    if n_qubits <= 1:
        return
    for ii in range(n_qubits):
        qml.CNOT(wires=[ii, (ii+1) % n_qubits])


def training_layer(n_qubits, params):
    for ii in range(n_qubits):
        rotations(ii, params[ii,:])
    entangle(n_qubits)


def encoding_layer(n_qubits, params):
    for ii in range(n_qubits):
        qml.RX(params[ii], wires=ii)


def enc_params(n_qubits):
    params = [np.random.uniform(-2*np.pi, np.pi) for _ in range(n_qubits)]
    return params

def rot_params(n_qubits, layers):
    params = np.empty((layers, n_qubits, global_rots), np.float64)
    for ll in range(layers):
        for nn in range(n_qubits):
            for aa in range(global_rots):
                params[ll, nn, aa] = np.random.uniform(-2*np.pi, np.pi)
    return params


@qml.qnode(device, diff_method="adjoint")
def circuit(n_qubits, layers, enc_params, rot_params, enc_layers):
    for ii in range(layers):
        if ii in enc_layers:
            encoding_layer(n_qubits, enc_params)
        training_layer(n_qubits, rot_params[ii])
    exp = qml.PauliZ(0)
    for ii in range(1, n_qubits):
        exp = exp @ qml.PauliZ(ii)
    return qml.expval(exp)


# Process target
def expectation(probs):
    # use expectation value to predict efficiency
    val = 0
    for key, value in probs.items():
        eig = 1
        for char in key:
            if char == "1":
                eig *= -1
        val += eig * value
    return val

def linmap(value, amin, amax, bmin, bmax):
    return bmin + (bmax - bmin)/(amax - amin) * (value - amin)

def polymap(value, amin, amax, bmin, bmax, power=2):
    m = linmap(value, amin, amax, 0, 1)
    m = m**power
    return linmap(m, 0, 1, bmin, bmax)

def expmap(value, amin, amax, bmin, bmax, base=np.exp(1)):
    m = linmap(value, amin, amax, 0, 1)
    m = base**m
    return linmap(m, 1, base, bmin, bmax)

def map_expval(expval, start, stop):
    return linmap(expval, -1, 1, start, stop)

# read data
def read_filtered(fname):
    data = pd.read_csv(fname)
    cols = ['TOP_INLET_ANGLE', 'BOTTOM_INLET_ANGLE', 'CAMBER_INLET_ANGLE', 'CAMBER_OUTLET_ANGLE', 'CAMBER_P1KFACTOR', 'CAMBER_P2X', 'CAMBER_P2Y', 'CAMBER_P3KFACTOR', 'PRESSURE LOSS']
    return data.dropna().drop_duplicates('PRESSURE LOSS')[cols]

data = read_filtered(infile)
# pick a subset of data to train on
cols = [
    'PRESSURE LOSS',
    #'TOP_INLET_ANGLE', # no effect
    'BOTTOM_INLET_ANGLE',
    #CAMBER_INLET_ANGLE', # no effect
    'CAMBER_OUTLET_ANGLE',
    'CAMBER_P1KFACTOR',
    'CAMBER_P2X',
    #'CAMBER_P2Y',
    #'CAMBER_P3KFACTOR'
]
d2 = data.drop(columns=cols).iloc[0:1]
data = data.merge(d2)[cols].to_numpy()

# target columns: target expectation value, input angles...
target = np.zeros((len(data), n_qubits+1))

# target mapping
adelta = 1
tdelta = 0.3
tmin = np.min(data[:,0])
tmax = np.max(data[:,0])
target[:,0] = linmap(data[:,0], tmin, tmax, -1 + tdelta, 1 - tdelta)
for c in range(1,len(cols)):
    amin = np.min(data[:,c])
    amax = np.max(data[:,c])
    target[:,c] = linmap(data[:,c], amin, amax, 0 + adelta, 2*np.pi - adelta)


# classical simulation routines
def simulate_efficency(params, n_qubits, layers, circuit):
    # classically simulate the circuit
    # set the parameter values using the inputs argument
    expval = circuit(n_qubits, layers, params[0], params[1])

    eff = map_expval(expval, 0, 1)

    return eff

def params_bounds(params_list):
    return [(0, 2 * np.pi) for _ in range(len(params_list))]


def new_tracker():
    tracker = {
        "count": 0,  # Elapsed optimization steps
        "cost": [],  # Cost at each step
        "params": [],  # Track parameters
        "time": [], # Step time
        "restart_step": 0
    }
    return tracker


init_params = rot_params(n_qubits, layers)

# pennylane optimisers
def train_opt(opt, params, target, circuit, n_qubits, layers, enc_layers, options, tracker, verbose=True):
    """Function to train VQE"""
    print("Starting the training.")

    print("=" * 80)
    print(f"OPTIMIZATION on {args.d} for {n_qubits} qubits, {layers} layers, {global_rots} rotations per set")
    print(f"Encoding layers {enc_layers}")

    if "maxiter" in options:
        print("Maximum iterations:", options["maxiter"])
    if "term" in options:
        print("Termination condition:", options["term"])

    print("Circuit configuration")
    print(qml.draw(circuit, max_length=140)(3, 2, [0, 0, 0], np.zeros((2,3,global_rots)), [0]))

    if not verbose:
        print('Param "verbose" set to False. Will not print intermediate steps.')
        print("=" * 80)

    def objective(params):
        angles = target[:, 1:].transpose()
        a = np.clip(circuit(n_qubits, layers, angles, params, enc_layers), -0.5, np.inf)
        b = np.clip(target[:,0], -0.5, np.inf)
        return np.mean(np.square(a - b))

    try:
        for i in range(options["maxiter"]):
            tracker.update({"count": tracker["count"] + 1})
            if verbose:
                print("=" * 80)
                print("Iteration step. Cycle:", tracker["count"])

            t1 = time.time()
            params, cost = opt.step_and_cost(objective, params)
            t2 = time.time()

            cost_delta = tracker["cost"][-1] - cost if i > 0 else None

            if verbose:
                print("cost:", cost)
                print("cost delta:", cost_delta)
                print("time:", t2-t1)

            # update tracker
            tracker["cost"].append(cost)
            tracker["params"].append(params)
            tracker["time"].append(t2-t1)

            # termination condition
            if "term" in options and options["term"] > cost:
                print("Termination condition reached at step:", i)
                break

            # vanishing gradient for past 20 iterations
            if reencode and i - tracker["restart_step"] > 20:
                mean_delta = np.mean(np.asarray(tracker["cost"][-6:-1]) - np.asarray(tracker["cost"][-5:]))
                if "grad_term" in options and mean_delta / cost < options["grad_term"]:
                    print("Vanishing gradient, adding encoding layer")
                    print(f"Mean cost delta of past 5 steps was {mean_delta}")
                    enc_layers.append(enc_layers[-1] + 1)
                    tracker.update({"restart_step": i})
                    print(f"Encoding layers {enc_layers} at step {tracker['restart_step']}")

    except KeyboardInterrupt:
        print("Keyboard interrupt, stopping training")

    # final run
    tracker["cost"].append(objective(params))
    tracker["params"].append(params)

    minid = np.argmin(tracker["cost"], requires_grad=False)
    cost = tracker["cost"][minid]
    params = tracker["params"][minid]
    print("Final cost:", cost)
    print("Final angles:", params)
    print("Training complete.")

    return cost, params, tracker


tracker1 = new_tracker()
opt = qml.AdamOptimizer(0.1)
fcost1, fparam1, tracker1 = train_opt(opt, init_params, target, circuit, n_qubits, layers, enc_layers, options, tracker1, verbose=verbose)
print("saving params to adam.npy")
np.save(f"{outdir}/adam", fparam1)
print("saving cost to cost.npy")
np.save(f"{outdir}/cost", np.asarray(tracker1["cost"]))


plt.figure()
color = "b"
plt.semilogy(tracker1["cost"], color=color)
plt.xlabel("step")
plt.ylabel("cost")
plt.yticks(color=color)

plt.twinx()
color = "r"
plt.plot(tracker1["time"], color=color)
plt.ylabel("time")
plt.yticks(color=color)
plt.tight_layout()
print("saving cost and runtime plot to adam.svg")
plt.savefig(f"{outdir}/adam.svg")


# run circuit with final parameters
angles = np.zeros((n_qubits, 2500))
x = np.linspace(0, 2*np.pi, 50)
y = np.linspace(0, 2*np.pi, 50)
X, Y = np.meshgrid(x,y)
angles[0, :] = X.flatten()
angles[1, :] = Y.flatten()

out = circuit(n_qubits, layers, angles, fparam1, enc_layers)
out = out.reshape(50,50)

# plotting
plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, out, cmap="jet")
#plt.plot(target[:,1], target[:,0])
ax.set_xlabel(cols[1])
ax.set_ylabel(cols[2])
ax.set_zlabel("output expectation value")

# plot differences
angles = target[:,1:].transpose()
a = np.clip(circuit(n_qubits, layers, angles, fparam1, enc_layers), -0.5, np.inf)
b = np.clip(target[:,0], -0.5, np.inf)
plt.figure()
plt.plot(a, alpha=0.5)
plt.plot(b, alpha=0.5)
plt.plot(a-b, alpha=0.5)
plt.legend(["qc", "target", "diff"])
plt.ylabel("error difference")
plt.tight_layout()
print("saving results plot to results.svg")
plt.savefig(f"{outdir}/results.svg")
