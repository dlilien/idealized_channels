# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from cartopolar import SPS
from cartopolar.antarctica_maps import stange, stange_channel, pig_full, pig_channel, STANGE_ASP, ANT_EXTENT
import matplotlib.gridspec
import matplotlib.colors as colors
import copy
import geopandas as gpd

crs = "EPSG:3031"

gl_fn = "../gl/measures_gl_lines.gpkg"
gl_ds = gpd.read_file(gl_fn).to_crs(crs)

width = 7.8
gs = matplotlib.gridspec.GridSpec(
    14,
    16,
    height_ratios=(0.01, 0.05, 0.85, 0.05, 0.1, 0.05, 0.95, 0.05, 0.2, 0.05, 0.3, 0.05, 0.26, 0.01),
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    wspace=0.0,
    width_ratios=(0.01, 0.23, 0.7, 0.02, 0.2, 0.02, 0.45, 0.02, 0.35, 0.1, 0.21, 0.11, 0.02, 0.59, 0.18, 0.01),
)
fig = plt.figure(figsize=(width, 5.2))

ax_ant = fig.add_subplot(gs[2:5, 11:14], projection=SPS())
ax_s1 = fig.add_subplot(gs[1:3, 2:5], projection=SPS())
ax_s2 = fig.add_subplot(gs[1:3, 6:10], projection=SPS())
ax_s1c = fig.add_subplot(gs[6:8, 8:12], projection=SPS())
ax_s2c = fig.add_subplot(gs[6:8, 13:15], projection=SPS())
ax_p1 = fig.add_subplot(gs[4:7, 2], projection=SPS())
ax_p2 = fig.add_subplot(gs[4:7, 4:7], projection=SPS())
ax_p1c = fig.add_subplot(gs[8:13, 1:4], projection=SPS())
ax_p2c = fig.add_subplot(gs[8:13, 4:9], projection=SPS())

ax_ant.set_xlim(ANT_EXTENT[:2])
ax_ant.set_ylim(ANT_EXTENT[2:])
ax_ant.show_tif(
    "../../antarctica_general/Quantarctica3/SatelliteImagery/MODIS/MODIS_Mosaic_1000m.tif", cmap="gray", vmin=0, vmax=70
)
ax_ant.set_facecolor("k")
ax_ant._y_inline = True
ax_ant.gridlines(
    xlocs=[0, 45, 90, 135, 180, -45, -90, -135],
    ylocs=[-60, -70, -80],
    color="0.6",
    draw_labels={"right": "x", "left": "x", "top": "x", "bottom": "x"},
)


stange(ax_s1)
stange(ax_s2)
# ax_s1c = fig.add_subplot(gs[4:7, 1:3], projection=SPS())
# ax_s2c = fig.add_subplot(gs[4:7, 4:7], projection=SPS())
stange_channel(ax_s1c)
stange_channel(ax_s2c)


ax_s1.gridlines(draw_labels={"bottom": "y", "left": "x"}, color="0.6", x_inline=False, y_inline=False)
ax_s2.gridlines(draw_labels={"bottom": "y"}, color="0.6", x_inline=False, y_inline=False)

# ax_p1 = fig.add_subplot(gs[6:8, 8:10], projection=SPS())
# ax_p2 = fig.add_subplot(gs[6:8, 11:14], projection=SPS())
pig_full(ax_p1)
pig_full(ax_p2)
pig_channel(ax_p1c)
pig_channel(ax_p2c)
ax_p1.gridlines(draw_labels={"bottom": "y", "left": "x"}, color="0.6", x_inline=False, y_inline=False)
ax_p2.gridlines(draw_labels={"bottom": "y"}, color="0.6", x_inline=False, y_inline=False)

v_fn = "../../antarctica_general/Quantarctica3/Glaciology/MEaSUREs Ice Flow Velocity/MEaSUREs_IceFlowSpeed_450m.tif"
s_fn_s = "../stange/rema_stange_100m.tif"
s_fn_p = "../pig/rema_pig_100m.tif"

surf_cm = copy.copy(plt.cm.terrain)
surf_cm.set_under("k")
cms = ax_s1.show_tif(s_fn_s, vmin=0, vmax=250, cmap=surf_cm)
cms = ax_s1c.show_tif(s_fn_s, vmin=0, vmax=250, cmap=surf_cm)
cms = ax_p1.show_tif(s_fn_p, vmin=0, vmax=250, cmap=surf_cm)
cms = ax_p1c.show_tif(s_fn_p, vmin=0, vmax=250, cmap=surf_cm)

norm = colors.LogNorm(10, 2000)
# norm = colors.Normalize(0, 1000)
v_cm = colors.LinearSegmentedColormap.from_list(
    "v", ["lightsalmon", "bisque", "gold", "olivedrab", "cornflowerblue", "mediumblue", "magenta", "red"], N=256
)
cmv = ax_s2.show_tif(v_fn, cmap=v_cm, norm=norm)
cmv = ax_s2c.show_tif(v_fn, cmap=v_cm, norm=norm)
cmv = ax_p2.show_tif(v_fn, cmap=v_cm, norm=norm)
cmv = ax_p2c.show_tif(v_fn, cmap=v_cm, norm=norm)

chan_color = "w"
stange_channel = gpd.read_file("../stange/stange_channel.gpkg").to_crs(crs)
for chan in stange_channel.iterrows():
    coords = np.array(chan[1].geometry.coords)
    if chan[0] == 0:
        lw = 0.5
    else:
        lw = 0.5
    # ax_s1.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
    ax_s2.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
    # ax_s1c.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
    ax_s2c.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
pig_channel = gpd.read_file("../pig/pig_side_channel.gpkg").to_crs(crs)
for chan in pig_channel.iterrows():
    if chan[1].geometry is not None:
        coords = np.array(chan[1].geometry.coords)
        if chan[0] == 0:
            lw = 0.5
        else:
            lw = 0.5
        # ax_p1.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
        ax_p2.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
        # ax_p1c.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)
        ax_p2c.plot(coords[:, 0], coords[:, 1], color=chan_color, lw=lw)

# cax_s1 = fig.add_subplot(gs[6, 0])
# cax_s2 = fig.add_subplot(gs[9, 0])
cax_s1 = fig.add_subplot(gs[9, 11:14])
cax_s2 = fig.add_subplot(gs[11, 11:14])
plt.colorbar(cmv, extend="both", label=r"Speed (m yr$^{-1}$)", cax=cax_s2, orientation="horizontal")
plt.colorbar(cms, extend="max", label=r"Surface elevation (m)", cax=cax_s1, orientation="horizontal")

for gll in gl_ds.iterrows():
    for gl in gll[1].geometry:
        coords = np.array(gl.coords)
        for ax in [ax_p1, ax_p2, ax_p1c, ax_p2c, ax_s1, ax_s2, ax_s1c, ax_s2c]:
            ax.plot(coords[:, 0], coords[:, 1], color="k")

ax_s1.set_facecolor("k")
ax_s2.set_facecolor("k")
ax_s1c.set_facecolor("k")
ax_s2c.set_facecolor("k")
ax_p1.set_facecolor("k")
ax_p2.set_facecolor("k")
ax_p1c.set_facecolor("k")
ax_p2c.set_facecolor("k")
for ax, letter in zip([ax_s1, ax_s2, ax_ant, ax_p1, ax_p2, ax_p1c, ax_p2c, ax_s1c, ax_s2c], "abcdefghijkl"):
    ax.text(
        0.01,
        0.99,
        letter,
        fontsize=14,
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="w", edgecolor="k", boxstyle="round", pad=0.1),
    )

ax_ant.show_other_ax(ax_s1, color="b")
ax_ant.show_other_ax(ax_p1, color="r")
fig.savefig("../plots/remote_sensing_channels.png", dpi=300)
