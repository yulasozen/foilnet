"""
Generate a Wigley hull as a closed (watertight) double-body STL for OpenFOAM.

Wigley half-breadth:  y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)
Double body: z spans [-T, +T] so the surface closes at top/bottom;
x spans [-L/2, +L/2] so it closes at bow/stern. -> watertight by construction.
Units: metres.
"""
import numpy as np
from stl import mesh

# Standard Wigley proportions
L = 6.0                 # length (m)
B = 0.10 * L            # beam
T = 0.0625 * L          # draft (half-depth of double body)

nx, nz = 120, 60        # surface resolution

xs = np.linspace(-L/2, L/2, nx)
zs = np.linspace(-T,  T,  nz)

def halfbreadth(x, z):
    return (B/2.0) * (1.0 - (2.0*x/L)**2) * (1.0 - (z/T)**2)

# Build a structured grid of points on the +y and -y surfaces, then triangulate.
def surface_points(sign):
    P = np.zeros((nx, nz, 3))
    for i, x in enumerate(xs):
        for k, z in enumerate(zs):
            y = sign * halfbreadth(x, z)
            P[i, k] = (x, y, z)
    return P

tris = []
def add_quad(p1, p2, p3, p4):
    tris.append([p1, p2, p3])
    tris.append([p1, p3, p4])

for sign, flip in ((+1, False), (-1, True)):
    P = surface_points(sign)
    for i in range(nx-1):
        for k in range(nz-1):
            a, b, c, d = P[i,k], P[i+1,k], P[i+1,k+1], P[i,k+1]
            if flip:   # keep outward normals consistent on the -y side
                add_quad(a, d, c, b)
            else:
                add_quad(a, b, c, d)

tris = np.array(tris)
m = mesh.Mesh(np.zeros(len(tris), dtype=mesh.Mesh.dtype))
m.vectors = tris
m.save('/home/claude/hullnet/wigley.stl')

# Report
mins = tris.reshape(-1,3).min(axis=0)
maxs = tris.reshape(-1,3).max(axis=0)
print(f"L={L}  B={B:.3f}  T(half)={T:.3f}")
print(f"triangles: {len(tris)}")
print(f"bbox min: {mins}")
print(f"bbox max: {maxs}")
