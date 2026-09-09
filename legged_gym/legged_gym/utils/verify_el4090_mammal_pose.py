#!/usr/bin/env python3
"""URDF forward-kinematics check for EL-4090 spider/mammal rest poses.

The script deliberately uses only the standard library.  It reads joint origins,
axes and limits directly from the URDF rather than assuming planar link lengths.
Run from the repository root (the optional height is the desired settled base
height above a level ground plane):

    python legged_gym/legged_gym/utils/verify_el4090_mammal_pose.py
    python legged_gym/legged_gym/utils/verify_el4090_mammal_pose.py --target-body-height 0.55
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
URDF = os.path.join(ROOT, "resources/robots/el_4090/urdf/el_4090.urdf")
LEGS = ("RF", "RM", "RB", "LF", "LM", "LB")

# The insect rest pose from El4090Envelop2Cfg.
SPIDER = {f"{leg}_{joint}": 0.0 for leg in LEGS for joint in ("HAA", "HFE", "KFE")}

# Mammal HAA angles encode the desired lateral leg orientation.  HFE/KFE are
# solved below to preserve the per-leg base-to-foot vertical offset of SPIDER.
MAMMAL_HAA = {
    "RF_HAA": -1.308, "RM_HAA": 1.308, "RB_HAA": 1.308,
    "LF_HAA": -1.308, "LM_HAA": 1.308, "LB_HAA": 1.308,
}
SPIDER_HAA = {f"{leg}_HAA": 0.0 for leg in LEGS}


def vec(text, default):
    return tuple(map(float, text.split())) if text else default


def eye():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def origin_transform(xyz, rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    # URDF's fixed-axis RPY convention: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rot = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
           [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
           [-sp, cp * sr, cp * cr]]
    out = eye()
    for i in range(3):
        out[i][:3] = rot[i]
        out[i][3] = xyz[i]
    return out


def axis_rotation(axis, angle):
    x, y, z = axis
    norm = math.sqrt(x*x + y*y + z*z)
    x, y, z = x/norm, y/norm, z/norm
    c, s, d = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    out = eye()
    out[0][:3] = [c+x*x*d, x*y*d-z*s, x*z*d+y*s]
    out[1][:3] = [y*x*d+z*s, c+y*y*d, y*z*d-x*s]
    out[2][:3] = [z*x*d-y*s, z*y*d+x*s, c+z*z*d]
    return out


def read_joints(urdf):
    joints = {}
    for node in ET.parse(urdf).getroot().findall("joint"):
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        joints[node.attrib["name"]] = {
            "parent": node.find("parent").attrib["link"], "child": node.find("child").attrib["link"],
            "xyz": vec(origin.attrib.get("xyz") if origin is not None else None, (0., 0., 0.)),
            "rpy": vec(origin.attrib.get("rpy") if origin is not None else None, (0., 0., 0.)),
            "axis": vec(axis.attrib.get("xyz") if axis is not None else None, (1., 0., 0.)),
            "limit": (float(limit.attrib["lower"]), float(limit.attrib["upper"])) if limit is not None else None,
            "type": node.attrib["type"],
        }
    return joints


def foot_position(joints, leg, angles):
    chain = [f"{leg}_HAA", f"{leg}_HFE", f"{leg}_KFE", f"{leg}_FIX"]
    transform = eye()
    for name in chain:
        joint = joints[name]
        transform = matmul(transform, origin_transform(joint["xyz"], joint["rpy"]))
        if joint["type"] in ("revolute", "continuous"):
            transform = matmul(transform, axis_rotation(joint["axis"], angles[name]))
    return tuple(transform[i][3] for i in range(3))


def solve_common_hfe_kfe(joints, target_z, haa, seed):
    """Coarse-to-fine bounded search, initialized near a natural bent leg."""
    hfe_range, kfe_range = joints["RF_HFE"]["limit"], joints["RF_KFE"]["limit"]
    best = None
    center_hfe, center_kfe = seed
    span = 0.8
    for step in (0.02, 0.002, 0.0002):
        half = max(span, step)
        hfe_lo, hfe_hi = max(hfe_range[0], center_hfe-half), min(hfe_range[1], center_hfe+half)
        kfe_lo, kfe_hi = max(kfe_range[0], center_kfe-half), min(kfe_range[1], center_kfe+half)
        hfe = hfe_lo
        while hfe <= hfe_hi + 1e-12:
            kfe = kfe_lo
            while kfe <= kfe_hi + 1e-12:
                pose = {**MAMMAL_HAA, **{f"{leg}_HFE": hfe for leg in LEGS}, **{f"{leg}_KFE": kfe for leg in LEGS}}
                # Match all legs and mildly prefer a compact, mammal-like bend.
                error = sum((foot_position(joints, leg, pose)[2] - target_z[leg])**2 for leg in LEGS)
                score = error + 1e-5 * ((hfe - seed[0])**2 + (kfe - seed[1])**2)
                if best is None or score < best[0]:
                    best = (score, error, hfe, kfe)
                kfe += step
            hfe += step
        _, _, center_hfe, center_kfe = best
        span = 2 * step
    return center_hfe, center_kfe


def make_pose(haa, hfe, kfe):
    return {
        **haa,
        **{f"{leg}_HFE": hfe for leg in LEGS},
        **{f"{leg}_KFE": kfe for leg in LEGS},
    }


def report_pose(joints, name, haa, target_z, seed):
    hfe, kfe = solve_common_hfe_kfe(joints, target_z, haa, seed)
    pose = make_pose(haa, hfe, kfe)
    print("{}: HFE={:.4f} rad, KFE={:.4f} rad".format(name, hfe, kfe))
    print("leg  target-foot-z [m]  solved-foot-z [m]  delta [mm]")
    for leg in LEGS:
        foot_z = foot_position(joints, leg, pose)[2]
        print("{:>2}   {:>+10.5f}       {:>+10.5f}    {:>+8.2f}".format(
            leg, target_z[leg], foot_z, (foot_z - target_z[leg]) * 1000
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-body-height",
        type=float,
        help="Solve both morphologies for this settled base height [m].",
    )
    args = parser.parse_args()
    joints = read_joints(URDF)
    print(f"URDF: {URDF}")
    if args.target_body_height is None:
        target_z = {leg: foot_position(joints, leg, SPIDER)[2] for leg in LEGS}
        print("Target: preserve the spider-zero rest height.")
        report_pose(joints, "mammal", MAMMAL_HAA, target_z, (1.0, -0.6))
        return
    if args.target_body_height <= 0.0:
        parser.error("--target-body-height must be positive")
    target_z = {leg: -args.target_body_height for leg in LEGS}
    print("Target settled body height: {:.4f} m".format(args.target_body_height))
    report_pose(joints, "spider", SPIDER_HAA, target_z, (0.0, 0.0))
    report_pose(joints, "mammal", MAMMAL_HAA, target_z, (0.2, -0.7))


if __name__ == "__main__":
    main()
