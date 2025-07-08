import numpy as np
import matplotlib.pyplot as plt
from main import de_casteljau

def compute_cubic_bezier_control_points(P0, t0, P3, t3, alpha, beta):
    t0 = t0 / np.linalg.norm(t0)
    t3 = t3 / np.linalg.norm(t3)

    P1 = P0 + alpha * t0
    P2 = P3 - beta * t3

    return np.array([P0, P1, P2, P3])

def de_casteljau_cubic(t, control_points):
    """Cubic Bézier via De Casteljau"""
    P = control_points
    A = (1 - t) * P[0] + t * P[1]
    B = (1 - t) * P[1] + t * P[2]
    C = (1 - t) * P[2] + t * P[3]
    D = (1 - t) * A + t * B
    E = (1 - t) * B + t * C
    point = (1 - t) * D + t * E
    return point


fig,axes = plt.subplots(nrows=2,sharex=False,sharey=False)
ax = axes[0]
fname = "./airfoil_camber_line.txt"
camber = np.loadtxt(fname)
fname = "./airfoil_bottom_thickness.txt"
thickness_bottom = np.loadtxt(fname)
fname = "./airfoil_top_thickness.txt"
thickness_top = np.loadtxt(fname)
ax.plot(camber[:,0],camber[:,1],"k--")

dy_dx = np.gradient(camber[:,1],camber[:,0])
# Normals to camber line
nx = -dy_dx / np.sqrt(1 + dy_dx**2)
ny = 1 / np.sqrt(1 + dy_dx**2)

x_upper = camber[:,0] + thickness_top[:,1] * nx
y_upper = camber[:,1] + thickness_top[:,1] * ny
x_lower = camber[:,0] - thickness_top[:,1] * nx
y_lower = camber[:,1] - thickness_top[:,1] * ny

ax.plot(x_lower,y_lower)
ax.plot(x_upper,y_upper)

for idx in [0,-1]:
    P0 = np.array([x_upper[idx], y_upper[idx]])
    P3 = np.array([x_lower[idx], y_lower[idx]])

    # Tangents
    dy_dx_upper = np.gradient(y_upper, x_upper)
    t0 = np.array([1.0, dy_dx_upper[idx]])

    dy_dx_lower = np.gradient(y_lower, x_lower)
    t3 = np.array([1.0, dy_dx_lower[idx]])

    radius = np.linalg.norm(P3 - camber[idx])  # or TE radius
    dist_best = 9999.9
    fact_best = 0.0
    for fact in np.linspace(0,2,100):
        alpha = beta = radius * fact  # adjust for curvature
        if idx == 0:
            fact_alpha = -1
            fact_beta = 1
        else:
            fact_alpha = 1
            fact_beta = -1
        control_points = compute_cubic_bezier_control_points(P0, t0, P3, t3, fact_alpha*alpha, fact_beta*beta)
        nt = 1000
        t_vals = np.linspace(0, 1, nt)
        # mid points #
        idx_mid = nt // 2 
        bezier_points = np.array([de_casteljau_cubic(t, control_points) for t in t_vals])
        dist_zero = np.sqrt((camber[idx,0]-x_lower[idx])**2+(camber[idx,1]-y_lower[idx])**2)
        dist_bez = np.sqrt((camber[idx,0]-bezier_points[idx_mid,0])**2+(camber[idx,1]-bezier_points[idx_mid,1])**2)

        dist_tmp = abs(dist_zero-dist_bez)
        if dist_tmp < dist_best:
            dist_best = dist_tmp
            fact_best = fact

    alpha = beta = radius * fact_best  # adjust for curvature
    if idx == 0:
        fact_alpha = -1
        fact_beta = 1
    else:
        fact_alpha = 1
        fact_beta = -1
    control_points = compute_cubic_bezier_control_points(P0, t0, P3, t3, fact_alpha*alpha, fact_beta*beta)
    nt = 1000
    t_vals = np.linspace(0, 1, nt)
    # mid points #
    idx_mid = nt // 2 
    bezier_points = np.array([de_casteljau_cubic(t, control_points) for t in t_vals])
    dist_zero = np.sqrt((camber[idx,0]-x_lower[idx])**2+(camber[idx,1]-y_lower[idx])**2)
    dist_bez = np.sqrt((camber[idx,0]-bezier_points[idx_mid,0])**2+(camber[idx,1]-bezier_points[idx_mid,1])**2)
    # ax.plot(bezier_points[:, 0], bezier_points[:, 1], 'k-')
    # for p in control_points:
    #     ax.scatter(p[0],p[1])
    ax.plot(control_points.T[0],control_points.T[1],color='r',marker='x')
    # ax.plot(camber[idx,0],camber[idx,1],"kx")
    # ax.plot(bezier_points[idx_mid,0],bezier_points[idx_mid,1],"kx")

    if idx == 0:
        le_arc_lower = bezier_points[idx_mid:,:]
        include_last_point = 1
        le_arc_upper = np.flip(bezier_points[:idx_mid+include_last_point,:],axis=0)
        camber = np.concatenate((bezier_points[idx_mid,:][np.newaxis,:],camber),axis=0)
    else:
        te_arc_lower = np.flip(bezier_points[idx_mid:,:],axis=0)
        include_last_point = 1
        te_arc_upper = bezier_points[:idx_mid+include_last_point,:]
        camber = np.concatenate((camber,bezier_points[idx_mid,:][np.newaxis,:]),axis=0)

# final blade #
# ax.plot(le_arc_lower[:,0],le_arc_lower[:,1])
# ax.plot(le_arc_upper[:,0],le_arc_upper[:,1])
# ax.plot(te_arc_lower[:,0],te_arc_lower[:,1])
# ax.plot(te_arc_upper[:,0],te_arc_upper[:,1])

# concatenate values #
x_lower = np.concatenate((le_arc_lower[:,0],x_lower,te_arc_lower[:,0]))
y_lower = np.concatenate((le_arc_lower[:,1],y_lower,te_arc_lower[:,1]))
x_upper = np.concatenate((le_arc_upper[:,0],x_upper,te_arc_upper[:,0]))
y_upper = np.concatenate((le_arc_upper[:,1],y_upper,te_arc_upper[:,1]))

ax = axes[1]
cax_dim = 50.0
# shift #
dx = camber[0,0]
dy = camber[0,1]
camber[:,0] -= dx
camber[:,1] -= dy
x_lower -= dx
x_upper -= dx
y_lower -= dy
y_upper -= dy
# rescale #
scale = camber[-1,0] / cax_dim
camber /= scale
x_lower /= scale
y_lower /= scale
x_upper /= scale
y_upper /= scale
ax.plot(camber[:,0],camber[:,1],"k--")
ax.plot(x_lower,y_lower,"r-")
ax.plot(x_upper,y_upper,"b-")

for ax in axes:
    ax.set_aspect('equal')

write_file = True
if write_file:
    fname_out = "./airfoil.dat"
    with open(fname_out,"w") as io_file:
        io_file.writelines(f"{cax_dim:.3f}\n")
        n_upper = x_upper.shape[0]
        n_lower = x_lower.shape[0]
        io_file.writelines(f"{n_upper}\n")
        for n in range(n_upper):
            io_file.writelines(f"{x_upper[n]:.15f} {y_upper[n]:.15f} \n")
        io_file.writelines(f"{n_lower}\n")
        for n in range(n_lower):
            io_file.writelines(f"{x_lower[n]:.15f} {y_lower[n]:.15f} \n")
    print("Wrote:",fname_out)

plt.show()
