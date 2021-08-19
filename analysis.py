from pymbar import bar
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

T = 298
R = 8.3144621E-3
J2cal = 1 / 4.184   

def rmse(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

def get_energy(lig, md_dir, deepmd_dir):
    # i - MM, j - DP
    ui_ri = np.loadtxt(os.path.join(md_dir, lig, "md_ener.xvg"), comments=["#", "@"])[:, 1]
    uj_rj = np.loadtxt(os.path.join(deepmd_dir, lig, "deepmd_ener.xvg"), comments=["#", "@"])[:, 1]
    uj_ri = np.loadtxt(os.path.join(deepmd_dir, lig, "deepmd_rerun_ener.xvg"), comments=["#", "@"])[:, 1]
    ui_rj = np.loadtxt(os.path.join(md_dir, lig ,"md_rerun_ener.xvg"), comments=["#", "@"])[:, 1]

    return ui_ri, uj_rj, uj_ri, ui_rj

def calc_diff_free_energy(energys, zero=0, unit='kcal'):
    ui_ri, uj_rj, uj_ri, ui_rj = energys
    assert ui_ri.size != 0
    assert uj_rj.size != 0
    assert uj_ri.size != 0
    assert ui_rj.size != 0
    # zero-point correction
    uj_rj -= zero
    uj_ri -= zero

    w_forward = (ui_rj - uj_rj) / (R * T)
    w_reverse = (uj_ri - ui_ri) / (R * T)

    # \Delta G(ML/MM) - \Delta G (MM)
    diff, std = bar.BAR(w_forward, w_reverse)
    diff, std = diff*R*T, std*R*T
    if unit == "kcal":
        diff, std = diff * J2cal, std * J2cal
    return diff, std

def calc_diff_free_energy_block_avg(energys, num_block=5, unit='kcal'):
    n = len(energys[0]) // num_block
    diffs = []
    stds = []
    for ii in range(num_block):
        if ii < (num_block - 1):
            eners = tuple(map(lambda arr: arr[ii * n: (ii+1) * n].copy(), energys))
        else:
            eners = tuple(map(lambda arr: arr[ii * n: ].copy(), energys))
        zero = np.mean(eners[2] - eners[0])
        diff, std = calc_diff_free_energy(eners, zero=zero, unit=unit)
        diffs.append(diff)
        stds.append(std)
    return np.mean(diffs), np.std(diffs)

def rsquared(x, y, degree=1):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot
    return results['determination']

def plot_corr(ori, corr, ori_std, corr_std, exp, output, simulation_time="500ps", unit='kcal'):
    ori_rmse = rmse(exp, ori)
    corr_rmse = rmse(exp, corr)
    ori_r2 = rsquared(exp, ori)
    corr_r2 = rsquared(exp, corr)


    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    ax.errorbar(exp, ori, ori_std, fmt='bo', label="origin")
    ax.errorbar(exp, corr, corr_std, fmt='go', label='corrected')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    rmin = min(xmin, ymin)
    rmax = max(xmax, ymax)
    diagline = np.linspace(rmin, rmax, 20)
    ax.fill_between(diagline, diagline - 1 * 2, diagline + 1 * 2, color='grey', alpha=0.2, zorder=0)
    ax.fill_between(diagline, diagline - 1 * 1, diagline + 1 * 1, color='grey', alpha=0.4, zorder=1)
    ax.plot(diagline, diagline, color='black', linewidth=3.)

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(rmin, rmax)
    ax.set_xlabel("$\Delta\Delta G_{\mathrm{exp}}$" + f" ({unit}/mol)")
    ax.set_ylabel("$\Delta\Delta G_{\mathrm{pred}}$" + f" ({unit}/mol)")
    ax.set_title(f"{simulation_time}, RMSE: {ori_rmse:.3f}/{corr_rmse: .3f}, $R^2$: {ori_r2:.2f}/{corr_r2:.2f}")
    ax.legend()
    fig.savefig(output, dpi=300)
    plt.close()
