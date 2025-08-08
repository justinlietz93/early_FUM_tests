# -*- coding: utf-8 -*-
# Interactive 3D FUM scaling with sliders (uses your FUM modules)
# Usage: python run_fum_scaling_3d_sliders.py
import sys, numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from FUM_Void_Equations import get_universal_constants, universal_void_dynamics
from FUM_Void_Debt_Modulation import VoidDebtModulation

UC = get_universal_constants()
ALPHA, BETA = UC['ALPHA'], UC['BETA']

def simulate_bundle(level_specs, T, params):
    rng = np.random.default_rng(params['seed'])
    modulator = VoidDebtModulation()
    mu = modulator.get_universal_domain_modulation(params['domain_key'])['domain_modulation']
    results = {}
    for lvl, label in level_specs:
        base_gamma = {0:1.30,1:1.20,2:1.10,3:1.05}[lvl] * params['gamma_scale']
        base_delta = {0:0.90,1:0.92,2:0.94,3:0.96}[lvl] * params['delta_scale']
        base_b     = {0:0.004,1:0.006,2:0.008,3:0.010}[lvl] * params['b_scale']
        paths = []
        for p in range(params['paths']):
            C = np.zeros_like(T); S = np.zeros_like(T); M = np.zeros_like(T); W = np.zeros_like(T)
            C[0] = 1e2; S[0] = (1e2 if lvl==0 else 5e2); M[0] = 5.0 + 2*lvl; W[0] = rng.uniform(0.2, 0.8)
            eta = ALPHA * mu; lam = BETA * mu
            jit_kC = 1.0 + rng.normal(0, 0.01); jit_b = 1.0 + rng.normal(0, 0.02)
            jit_eta = 1.0 + rng.normal(0, 0.02); jit_lam = 1.0 + rng.normal(0, 0.02)
            for t in range(1, len(T)):
                C[t] = C[t-1] * (1 + params['kappa_C'] * jit_kC)
                dW = universal_void_dynamics(W[t-1], t, domain_modulation=mu, use_time_dynamics=params['use_time_dynamics'])
                W[t] = np.clip(W[t-1] + dW, 0.0, 1.0)
                if lvl == 0:
                    S[t] = S[t-1] * (1 + 0.06 * jit_b)
                else:
                    S[t] = S[t-1] + (base_b * jit_b) * (max(M[t-1], 0.0) ** base_delta)
                L = max(dW, 0.0)
                M[t] = M[t-1] + (eta * jit_eta) * (max(S[t-1], 1.0) ** base_gamma) * (1.0 + L) - (lam * jit_lam) * M[t-1]
            paths.append((C,S,M))
        results[lvl] = dict(label=label, paths=paths)
    return results

def baseline_M(C, B0):
    return 0.0 + B0*np.log10(C)

def decade_ticks(vmin, vmax):
    start = int(np.floor(vmin)); end = int(np.ceil(vmax))
    ticks = np.arange(start, end + 1); labels = [r"$10^{%d}$" % k for k in ticks]
    return ticks, labels

def build_interactive():
    T = np.linspace(1, 120, 160)
    level_specs = [(0,"Foundational (Trunk)"),(1,"Domain Mastery"),(2,"Sub-Domain Mastery"),(3,"Micro-Specialization")]
    params = dict(domain_key='standard_model', gamma_scale=1.00, delta_scale=1.00, b_scale=1.00, kappa_C=0.12,
                  baseline_slope=0.07, paths=6, seed=7, use_time_dynamics=True)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.06, right=0.80, bottom=0.20)

    results = simulate_bundle(level_specs, T, params)
    line_groups = {}
    for lvl, _ in level_specs:
        lines = []
        for (C,S,M) in results[lvl]['paths']:
            l, = ax.plot(np.log10(C), np.log10(S), M, linewidth=1.4)
            lines.append(l)
        line_groups[lvl] = lines

    Cb = np.geomspace(1e2, 1e14, 200)
    Mb = baseline_M(Cb, params['baseline_slope'])
    (baseline_line,) = ax.plot(np.log10(Cb), np.log10(Cb**0.25), Mb, linestyle='--', linewidth=2)

    metric_text = ax.text2D(0.82, 0.92, "", transform=ax.transAxes, ha='left', va='top')

    xmin, xmax = np.log10(1e2), np.log10(1e14)
    ymin, ymax = np.log10(1e2), np.log10(1e10)
    xt, xl = decade_ticks(xmin, xmax); yt, yl = decade_ticks(ymin, ymax)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_xlabel('Compute (FLOPs, log10)'); ax.set_ylabel('Scale (Tasks/Scope, log10)'); ax.set_zlabel('Time / Cumulative Mastery')
    ax.view_init(elev=20, azim=-55)

    ax_kC    = plt.axes([0.10, 0.10, 0.25, 0.03])
    ax_gam   = plt.axes([0.10, 0.06, 0.25, 0.03])
    ax_b     = plt.axes([0.47, 0.10, 0.25, 0.03])
    ax_del   = plt.axes([0.47, 0.06, 0.25, 0.03])
    ax_base  = plt.axes([0.10, 0.02, 0.25, 0.03])
    ax_paths = plt.axes([0.47, 0.02, 0.25, 0.03])

    s_kC    = Slider(ax_kC,   'κ_C (compute)', 0.02, 0.25, valinit=params['kappa_C'], valstep=0.005)
    s_gamma = Slider(ax_gam,  'γ scale',       0.90, 1.30, valinit=params['gamma_scale'], valstep=0.01)
    s_b     = Slider(ax_b,    'b scale',       0.5,  2.0,  valinit=params['b_scale'],    valstep=0.05)
    s_delta = Slider(ax_del,  'δ scale',       0.85, 1.05, valinit=params['delta_scale'], valstep=0.01)
    s_base  = Slider(ax_base, 'LLM slope b',   0.03, 0.12, valinit=params['baseline_slope'], valstep=0.005)
    s_paths = Slider(ax_paths,'paths',         1, 10, valinit=params['paths'], valstep=1)

    ax_radio = plt.axes([0.83, 0.55, 0.15, 0.25])
    radio = RadioButtons(ax_radio, ('Mod 1','Mod 2','Mod 3','Mod 4'), active=1)
    ax_chk = plt.axes([0.83, 0.50, 0.15, 0.08])
    chk = CheckButtons(ax_chk, ['time dynamics'], [params['use_time_dynamics']])

    def baseline(C): return baseline_M(C, s_base.val)

    def refresh_metrics(res):
        lines = []
        for lvl, _ in level_specs:
            C, S, M = res[lvl]['paths'][0]
            Cfinal, Mfinal = C[-1], M[-1]
            Mb_final = baseline(np.array([Cfinal]))[0]
            lines.append(f"L{lvl}: ΔM={Mfinal - Mb_final: .2f}")
        metric_text.set_text("Lift vs. LLM @ final C:\n" + "\n".join(lines))

    def update(_):
        params['kappa_C']     = s_kC.val
        params['gamma_scale'] = s_gamma.val
        params['b_scale']     = s_b.val
        params['delta_scale'] = s_delta.val
        params['baseline_slope'] = s_base.val
        params['paths'] = int(s_paths.val)
        params['domain_key'] = radio.value_selected
        params['use_time_dynamics'] = chk.get_status()[0]

        res = simulate_bundle(level_specs, T, params)

        for lvl, _ in level_specs:
            lines = line_groups[lvl]
            if len(lines) != len(res[lvl]['paths']):
                for line in lines: line.remove()
                new_lines = []
                for (C,S,M) in res[lvl]['paths']:
                    l, = ax.plot(np.log10(C), np.log10(S), M, linewidth=1.4)
                    new_lines.append(l)
                line_groups[lvl] = new_lines
            else:
                for i, (C,S,M) in enumerate(res[lvl]['paths']):
                    lines[i].set_data_3d(np.log10(C), np.log10(S), M)

        Cb = np.geomspace(1e2, 1e14, 200); Mb = baseline(Cb)
        baseline_line.set_data_3d(np.log10(Cb), np.log10(Cb**0.25), Mb)

        refresh_metrics(res)
        fig.canvas.draw_idle()

    for s in (s_kC, s_gamma, s_b, s_delta, s_base, s_paths): s.on_changed(update)
    radio.on_clicked(update); chk.on_clicked(update)

    refresh_metrics(results)
    plt.show()

if __name__ == '__main__':
    build_interactive()
