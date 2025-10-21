# coding: utf-8

# In[1]:


# general imports
import time
import argparse
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

from scipy.optimize import minimize


parser = argparse.ArgumentParser()
parser.add_argument("-i", default="multall_runs.csv")
parser.add_argument("-o", default=".")
parser.add_argument("-v", action="store_true")

args = parser.parse_args()
infile = args.i
outdir = args.o
verbose = args.v

# In[2]:


n_qubits = 6
layers = 6
device = qml.device("lightning.qubit", wires=n_qubits)
options = {"maxiter": 100}


# In[3]:


global_rots = 2
def rotations(wire, params):
    qml.RZ(params[0], wires=wire)
    qml.RY(params[1], wires=wire)
    #qml.RZ(params[2], wires=wire)

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


# In[4]:


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


# In[5]:


@qml.qnode(device, diff_method="adjoint")
def circuit(n_qubits, layers, enc_params, rot_params):
    for ii in range(layers):
        encoding_layer(n_qubits, enc_params)
        training_layer(n_qubits, rot_params[ii])
    exp = qml.PauliZ(0)
    for ii in range(1, n_qubits):
        exp = exp @ qml.PauliZ(ii)
    return qml.expval(exp)


# In[6]:


# In[7]:


# In[8]:


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

def map_expval(expval, start, stop):
    return linmap(expval, -1, 1, start, stop)


# In[9]:

# In[10]:


def simulate_efficency(params, n_qubits, layers, circuit):
    # classically simulate the circuit
    # set the parameter values using the inputs argument
    expval = circuit(n_qubits, layers, params[0], params[1])

    eff = map_expval(expval, 0, 1)

    return eff


# In[11]:


def objective_function(params, target, circuit, n_qubits, layers, tracker, verbose):
    tracker.update({"count": tracker["count"] + 1})
    if verbose:
        print("=" * 80)
        print("Iteration step. Cycle:", tracker["count"])

    # reshape params
    params = params.reshape((layers, n_qubits, global_rots))

    def cost(params):
        angles = target[:,0]
        angles = np.tile(angles, (n_qubits,1))
        return np.mean(np.square(circuit(n_qubits, layers, angles, params) - target[:,1]))

    # minimize MSE with target dataset
    mse = cost(params)

    if verbose:
        print("MSE:", mse)

    # update tracker
    tracker["error"].append(mse)
    tracker["params"].append(params)

    return mse


# In[12]:


def train(func, params, target, circuit, n_qubits, layers, options, tracker, opt_method="cobyla", verbose=True):
    """Function to train VQE"""
    print("Starting the training.")

    print("=" * 80)
    print(f"OPTIMIZATION for {n_qubits} qubits, {layers} layers")

    if not verbose:
        print('Param "verbose" set to False. Will not print intermediate steps.')
        print("=" * 80)

    # parameter bounds
    params = params.flatten()
    bounds = params_bounds(params)

    # run classical optimization (example: method='Nelder-Mead')
    result = minimize(
        func,
        params,
        args=(target, circuit, n_qubits, layers, tracker, verbose),
        bounds=bounds,
        options=options,
        method=opt_method,
    )

    # store result of classical optimization
    # store result of classical optimization
    cost = result.fun
    print("Final cost:", cost)
    result_angles = result.x
    print("Final angles:", result_angles)
    print("Training complete.")

    return cost, params, tracker

def params_bounds(params_list):
    return [(0, 2 * np.pi) for _ in range(len(params_list))]


# In[13]:


target = np.loadtxt(infile, delimiter=",", usecols=(3,12), skiprows = 1)
adelta = 1.5
tdelta = 0.1
amin = np.min(target[:,0])
amax = np.max(target[:,0])
tmin = np.min(target[:,1])
tmax = np.max(target[:,1])
for ii in range(target.shape[0]):
    target[ii,0] = linmap(target[ii,0], amin, amax, 0 + adelta, 2*np.pi - adelta)
    target[ii,1] = linmap(target[ii,1], tmin, tmax, -1 + tdelta, 1 - tdelta)


# In[14]:


def new_tracker():
    tracker = {
        "count": 0,  # Elapsed optimization steps
        "error": [],  # Error at each step
        "params": [],  # Track parameters
        "time": [], # Step time
    }
    return tracker


# In[15]:


# set tracker to keep track of results
tracker = new_tracker()
init_params = rot_params(n_qubits, layers)
#fcost, fparam, tracker = train(objective_function, init_params, target, circuit, n_qubits, layers, options, tracker, verbose=False)
#np.save("cobyla", fparam)

# In[16]:


plt.figure()
plt.plot(tracker["error"])
#plt.savefig("cobyla.svg")


# In[17]:


def train_opt(opt, params, target, circuit, n_qubits, layers, options, tracker, verbose=True):
    """Function to train VQE"""
    print("Starting the training.")

    print("=" * 80)
    print(f"OPTIMIZATION for {n_qubits} qubits, {layers} layers")

    if not verbose:
        print('Param "verbose" set to False. Will not print intermediate steps.')
        print("=" * 80)

    def cost(params):
        angles = target[:,0]
        angles = np.tile(angles, (n_qubits,1))
        return np.mean(np.square(circuit(n_qubits, layers, angles, params) - target[:,1]))

    for i in range(options["maxiter"]):
        tracker.update({"count": tracker["count"] + 1})
        if verbose:
            print("=" * 80)
            print("Iteration step. Cycle:", tracker["count"])

        t1 = time.time()
        params, mse = opt.step_and_cost(cost, params)
        t2 = time.time()

        if verbose:
            print("MSE:", mse)

        # update tracker
        tracker["error"].append(mse)
        tracker["params"].append(params)
        tracker["time"].append(t2-t1)

    # final run
    tracker["error"].append(cost(params))
    tracker["params"].append(params)

    minid = np.argmin(tracker["error"], requires_grad=False)
    cost = tracker["error"][minid]
    params = tracker["params"][minid]
    print("Final cost:", cost)
    print("Final angles:", params)
    print("Training complete.")

    return cost, params, tracker


# In[18]:


tracker1 = new_tracker()
opt = qml.AdamOptimizer(0.1)
fcost1, fparam1, tracker1 = train_opt(opt, init_params, target[::5, :], circuit, n_qubits, layers, options, tracker1, verbose=verbose)
np.save(f"{outdir}/adam", fparam1)

# In[19]:


plt.figure()
color = "b"
plt.plot(tracker1["error"], color=color)
plt.xlabel("step")
plt.ylabel("error")
plt.yticks(color=color)

plt.twinx()
color = "r"
plt.plot(tracker1["time"], color=color)
plt.ylabel("time")
plt.yticks(color=color)
plt.tight_layout()
plt.savefig(f"{outdir}/adam.svg")


# run circuit with final parameters
angles = np.linspace(0, 2*np.pi, 100)
angles = np.tile(angles, (n_qubits,1))
#out = circuit(n_qubits, layers, angles, fparam.reshape(layers, n_qubits, global_rots))
out1 = circuit(n_qubits, layers, angles, fparam1)

plt.figure()
#plt.plot(angles[0,:], out)
plt.plot(angles[0,:], out1)
plt.plot(target[:,0], target[:,1])
plt.xlabel("input angle")
plt.ylabel("output expectation value")
plt.savefig(f"{outdir}/results.svg")

