#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 dlilien <dlilien@noatak.local>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np
from libchannels import ts_name
from hybrid_channels import widths, depths

depths = np.array(depths)
widths = np.array(widths)

setup = ts_name
names = ["full_chans", "full_even", "inner_chans", "inner_even", "outer_chans", "outer_even", "margin_chans", "margin_even", "funky_chans", "funky_even", "center_chans", "center_even"]

cache_name = "pointwise_partial_stream.npz"
with np.load(cache_name) as data:
    for name in names:
        exec("""ts_{:s} = data['{:s}']""".format(name, name))

ts_dict = {
    "full": {"center": [ts_full_chans[0, :, :], ts_full_even[0, :, :]], "gl": [ts_full_chans[3, :, :], ts_full_even[3, :, :]]},
    "funky": {"center": [ts_funky_chans[0, :, :], ts_funky_even[0, :, :]], "gl": [ts_funky_chans[3, :, :], ts_funky_even[3, :, :]]},
    "margin": {"center": [ts_margin_chans[0, :, :], ts_margin_even[0, :, :]], "gl": [ts_margin_chans[3, :, :], ts_margin_even[3, :, :]]},
    "center": {"center": [ts_center_chans[0, :, :], ts_center_even[0, :, :]], "gl": [ts_center_chans[3, :, :], ts_center_even[3, :, :]]},
    "inner": {"center": [ts_inner_chans[0, :, :], ts_inner_even[0, :, :]], "gl": [ts_inner_chans[3, :, :], ts_inner_even[3, :, :]]},
    "outer": {"center": [ts_outer_chans[0, :, :], ts_outer_even[0, :, :]], "gl": [ts_outer_chans[3, :, :], ts_outer_even[3, :, :]]},
}

names = [
    "full_center_vels_chans",
    "full_center_vels_even",
    "full_outer_vels_chans",
    "full_outer_vels_even",
    "full_gl_vels_chans",
    "full_gl_vels_even",
    "full_ugl_vels_chans",
    "full_ugl_vels_even",
    "inner_center_vels_chans",
    "inner_center_vels_even",
    "inner_outer_vels_chans",
    "inner_outer_vels_even",
    "inner_gl_vels_chans",
    "inner_gl_vels_even",
    "inner_ugl_vels_chans",
    "inner_ugl_vels_even",
    "outer_center_vels_chans",
    "outer_center_vels_even",
    "outer_outer_vels_chans",
    "outer_outer_vels_even",
    "outer_gl_vels_chans",
    "outer_gl_vels_even",
    "outer_ugl_vels_chans",
    "outer_ugl_vels_even",
    "margin_center_vels_chans",
    "margin_center_vels_even",
    "margin_outer_vels_chans",
    "margin_outer_vels_even",
    "margin_gl_vels_chans",
    "margin_gl_vels_even",
    "margin_ugl_vels_chans",
    "margin_ugl_vels_even",
    "center_center_vels_chans",
    "center_center_vels_even",
    "center_outer_vels_chans",
    "center_outer_vels_even",
    "center_gl_vels_chans",
    "center_gl_vels_even",
    "center_ugl_vels_chans",
    "center_ugl_vels_even",
]

cache_name = "pointwise_mismip.npz"
with np.load(cache_name) as data:
    for name in names:
        exec("{:s} = data['{:s}']".format(name, name))

mismip_dict = {
    "full": {"center": [full_center_vels_chans[:, :], full_center_vels_even[:, :]], "gl": [full_gl_vels_chans[:, :], full_gl_vels_even[:, :]]},
    "margin": {"center": [margin_center_vels_chans[:, :], margin_center_vels_even[:, :]], "gl": [margin_gl_vels_chans[:, :], margin_gl_vels_even[:, :]]},
    "funky": {"center": [margin_center_vels_chans[:, :], margin_center_vels_even[:, :]], "gl": [margin_gl_vels_chans[:, :], margin_gl_vels_even[:, :]]},
    "center": {"center": [center_center_vels_chans[:, :], center_center_vels_even[:, :]], "gl": [center_gl_vels_chans[:, :], center_gl_vels_even[:, :]]},
    "inner": {"center": [inner_center_vels_chans[:, :], inner_center_vels_even[:, :]], "gl": [inner_gl_vels_chans[:, :], inner_gl_vels_even[:, :]]},
    "outer": {"center": [outer_center_vels_chans[:, :], outer_center_vels_even[:, :]], "gl": [outer_gl_vels_chans[:, :], outer_gl_vels_even[:, :]]},
}

alldiffs = {pos: [] for pos in mismip_dict["full"]}
print("Relative difference (ts - mismip) / mismip in percent")
for sim in mismip_dict:
    if sim in ["outer", "center"]:
        continue
    md, td = mismip_dict[sim], ts_dict[sim]
    for pos in md:
        [mc, me] = md[pos]
        [tc, te] = td[pos]
        reldiff = ((tc - te) - (mc - me)) / (mc - me) * 100.0
        alldiffs[pos].append(reldiff)
        print("For {:s} at {:s}, diff is {:4.1f}±{:4.1f}%".format(sim, pos, np.mean(reldiff), np.std(reldiff)))
for pos, alldiff in alldiffs.items():
    print("Overall, at {:s}, diff is {:4.1f}±{:4.1f}%".format(pos, np.mean(np.hstack(alldiff)), np.std(np.hstack(alldiff))))

for pos in alldiffs:
    [fc, fe] = ts_dict["funky"][pos]
    [mc, me] = ts_dict["margin"][pos]
    reldiff = ((fc - fe) - (mc - me)) / (mc - me) * 100.0
    print("Comparing left and right marginal channels at {:s}, for left flow is {:4.1f}±{:4.1f}% faster".format(pos, np.mean(reldiff), np.std(reldiff)))
