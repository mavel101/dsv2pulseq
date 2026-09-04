"""Computes which physical gradient channel (x/y/z) and polarity each logical
(readout/phase/slice) channel drives, directly from a Siemens .pro protocol
dump's slice orientation (sNormal + dInPlaneRot) -- no dependence on any
gradient-waveform .dsv file.

Ported from Koma_simulation/pro_params.py, itself a direct port of Siemens'
own IDEA sequence library (fGSLCalcPRS.cpp / fGSLClassOri.cpp, as
reimplemented in https://github.com/wtclarke/spec2nii's
spec2nii/GSL/gslfunctions.py), mapped from the patient coordinate system
(PCS: Sag/Cor/Tra) to the physical gradient/device coordinate system
(DCS: GX/GY/GZ) assuming head-first-supine patient positioning.
"""
import re

import numpy as np

AXES = ("x", "y", "z")
# PCS (Sag,Cor,Tra) -> DCS (GX,GY,GZ) for head-first-supine.
PCS_TO_DCS_HFS = np.diag([1.0, -1.0, -1.0])


def parse_pro(path):
    """Flat {dotted.key: value_str} dict from a .pro file's ASCCONV-style lines
    (e.g. "sKSpace.lBaseResolution\t = \t134"). Ignores the surrounding nested
    XProtocol wrapper (UI/EVAStringTable sections etc.), which doesn't match
    this pattern."""
    params = {}
    pattern = re.compile(r'^([A-Za-z_][\w.\[\]]*)\s*=\s*(.+?)\s*$')
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                params[m.group(1)] = m.group(2)
    return params


def class_ori(sag, cor, tra):
    """Port of IDEA-VB17/n4/pkg/MrServers/MrMeasSrv/SeqFW/libGSL/fGSLClassOri.cpp.
    Returns 0=Sagittal, 1=Coronal, 2=Transverse."""
    abs_sag, abs_cor, abs_tra = abs(sag), abs(cor), abs(tra)
    eq_sag_cor = np.isclose(abs_sag, abs_cor)
    eq_sag_tra = np.isclose(abs_sag, abs_tra)
    eq_cor_tra = np.isclose(abs_cor, abs_tra)
    if ((eq_sag_cor and eq_sag_tra) or
            (eq_sag_cor and abs_sag < abs_tra) or
            (eq_sag_tra and abs_sag > abs_cor) or
            (eq_cor_tra and abs_cor > abs_sag) or
            (abs_sag > abs_cor and abs_sag < abs_tra) or
            (abs_sag < abs_cor and abs_cor < abs_tra) or
            (abs_sag < abs_tra and abs_tra > abs_cor) or
            (abs_cor < abs_tra and abs_tra > abs_sag)):
        return 2  # Transverse
    if ((eq_sag_cor and abs_sag > abs_tra) or
            (eq_sag_tra and abs_sag < abs_cor) or
            (abs_sag < abs_cor and abs_cor > abs_tra) or
            (abs_sag > abs_tra and abs_sag < abs_cor) or
            (abs_sag < abs_tra and abs_tra < abs_cor)):
        return 1  # Coronal
    if ((eq_cor_tra and abs_cor < abs_sag) or
            (abs_sag > abs_cor and abs_sag > abs_tra) or
            (abs_cor > abs_tra and abs_cor < abs_sag) or
            (abs_cor < abs_tra and abs_tra < abs_sag)):
        return 0  # Sagittal
    raise ValueError("Invalid slice orientation")


def calc_prs(gs, phi):
    """Port of IDEA-VB17/n4/pkg/MrServers/MrMeasSrv/SeqFW/libGSL/fGSLCalcPRS.cpp.
    Phase (gp) and read (gr) direction vectors [[PCS]] from the slice normal gs
    and in-plane rotation phi [rad]."""
    orientation = class_ori(gs[0], gs[1], gs[2])
    gp = np.zeros(3)
    if orientation == 2:  # Transverse
        gp[1] = gs[2] / np.sqrt(gs[1] ** 2 + gs[2] ** 2)
        gp[2] = -gs[1] / np.sqrt(gs[1] ** 2 + gs[2] ** 2)
    elif orientation == 1:  # Coronal
        gp[0] = gs[1] / np.sqrt(gs[0] ** 2 + gs[1] ** 2)
        gp[1] = -gs[0] / np.sqrt(gs[0] ** 2 + gs[1] ** 2)
    else:  # Sagittal
        gp[0] = -gs[1] / np.sqrt(gs[0] ** 2 + gs[1] ** 2)
        gp[1] = gs[0] / np.sqrt(gs[0] ** 2 + gs[1] ** 2)
    gr = np.cross(gs, gp)
    gp = np.cos(phi) * gp - np.sin(phi) * gr
    gr = np.cross(gs, gp)
    return gp, gr


def dominant_axis(vec):
    """(axis name, sign) of a vector's dominant physical-channel component."""
    idx = int(np.argmax(np.abs(vec)))
    sign = 1 if vec[idx] >= 0 else -1
    return AXES[idx], sign


def channel_map_from_pro(path):
    """
    Returns {'r': (axis, sign), 'p': (axis, sign), 's': (axis, sign)}, mapping
    each logical gradient channel (readout/phase/slice) to the physical
    channel (x/y/z) and the polarity ('physical amplitude' = sign * 'declared
    .INF amplitude') it drives, for this .pro file's slice orientation.
    """
    params = parse_pro(path)

    def get(key, default=0.0):
        return float(params[key]) if key in params else default

    sag = get("sSliceArray.asSlice[0].sNormal.dSag")
    cor = get("sSliceArray.asSlice[0].sNormal.dCor")
    tra = get("sSliceArray.asSlice[0].sNormal.dTra")
    inplane_rot = get("sSliceArray.asSlice[0].dInPlaneRot")

    gs = np.array([sag, cor, tra])
    # Unlike Koma_simulation/pro_params.py (which negates dInPlaneRot -- verified
    # empirically there for the *reconstructed-image* orientation convention),
    # raw hardware gradient polarity needs dInPlaneRot un-negated: verified
    # empirically against real GRX/GRY samples for a 90-degree-rotated sequence
    # (miFB_15_z0_RL), where negating it gives the wrong sign for both the
    # readout and phase channels. dInPlaneRot=0 makes this a no-op either way,
    # which is why it went unnoticed for non-rotated sequences.
    gp, gr = calc_prs(gs, inplane_rot)

    gr_dcs = PCS_TO_DCS_HFS @ gr
    gp_dcs = PCS_TO_DCS_HFS @ gp
    gs_dcs = PCS_TO_DCS_HFS @ gs

    return {
        'r': dominant_axis(gr_dcs),
        'p': dominant_axis(gp_dcs),
        's': dominant_axis(gs_dcs),
    }
