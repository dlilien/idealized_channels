# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from icepackaccs import friction
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

u = np.linspace(0, 1000, 250)

fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(7.8, 5), gridspec_kw={"height_ratios": (1, 0.8, 0.8)})


def effP(s, h):
    p_W = ρ_W * g * np.maximum(-(s - h), 0)
    p_I = ρ_I * g * h
    return np.maximum(p_I - p_W, 0)


def surf(N, h):
    return (N - ρ_I * g * h + ρ_W * g * h) / (ρ_W * g)


for i, N in enumerate([0.05, 0.1, 0.5, 1.0]):
    s = surf(N, 1000)
    ax.plot(
        u,
        1.0e3 * friction.tau_mismip_assaydavis(3.0, u, 1000, s, 0.5),
        color=f"C{i}",
        label="N={:d} kPa".format(int(N * 1000)),
    )
    ax.plot(
        u,
        1.0e3 * friction.tau_regularized_coulomb_mismip(3.0, u, 1000, s, 0.5),
        color=f"C{i}",
        linestyle="dashed",
    )
    ax1.plot(
        u,
        1.0e3
        * (
            friction.tau_regularized_coulomb_mismip(3.0, u, 1000, s, 0.5)
            - friction.tau_mismip_assaydavis(3.0, u, 1000, s, 0.5)
        ),
        color=f"C{i}",
    )
    ax2.plot(
        u,
        100
        * (
            friction.tau_regularized_coulomb_mismip(3.0, u, 1000, s, 0.5)
            - friction.tau_mismip_assaydavis(3.0, u, 1000, s, 0.5)
        )
        / friction.tau_mismip_assaydavis(3.0, u, 1000, s, 0.5),
        color=f"C{i}",
    )

ax.plot([], [], color="k", label="Asay-Davis")
ax.plot([], [], color="k", label="Eq. 4", linestyle="dashed")
ax.legend(loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.05, 1.05))
ax2.set_xlabel(r"Speed (m a$^{-1})$")
ax1.set_ylabel("Absolute\ndifference (kPa)")
ax2.set_ylabel("Relative\ndifference (%)")
ax.set_ylabel("Basal shear\nstress (kPa m$^{-2}$)")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 125)
ax1.set_ylim(0, 3)
ax2.set_ylim(0, 7.5)

for a, l in zip([ax, ax1, ax2], "abcd"):
    a.text(0.005, 0.995, l, ha="left", va="top", fontsize=14, transform=a.transAxes)

fig.tight_layout(pad=0.5)
fig.savefig("../plots/coulomb_sliding_diff.pdf")
