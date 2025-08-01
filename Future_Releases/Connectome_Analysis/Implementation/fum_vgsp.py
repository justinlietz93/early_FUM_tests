# fum_vgsp.py
# THIS CODE IS AN EARLY STEP IN THE DEMO PROCESS
# The most up to date version is resonance_enhanced_vgsp.py

import numpy as np
from scipy.sparse import csc_matrix, find

def apply_vgsp_updates(W, spike_times, time_step, eta, mod_factor, lambda_decay, params, is_excitatory):
    """
    Applies Valence-Gated Synaptic Plasticity (VGSP) to the weight matrix.
    
    This version correctly implements the stability mechanisms identified in the
    FUM stability analysis documentation:
    1.  Calculates a CRET-potential (Hebbian term).
    2.  Modulates it with the squashed reward signal (mod_factor).
    3.  Applies a linear weight decay (`lambda_decay`).
    4.  Enforces strict E/I weight constraints.
    """
    # Get vgsp hyperparameters
    A_plus = params.get('A_plus', 0.1)
    A_minus = params.get('A_minus', 0.12)
    tau_plus = params.get('tau_plus', 20.0)
    tau_minus = params.get('tau_minus', 20.0)
    time_window = 250 # ms

    # Find all existing synaptic pathways
    rows, cols, _ = find(W)
    plasticity_impulse = np.zeros(W.shape)

    recent_spikes = [[s for s in st if s > (time_step - time_window)] for st in spike_times]

    for i in range(len(rows)):
        pre_idx, post_idx = cols[i], rows[i]
        
        recent_pre = recent_spikes[pre_idx]
        recent_post = recent_spikes[post_idx]

        if not recent_pre or not recent_post:
            continue

        delta_t_matrix = np.subtract.outer(recent_post, recent_pre)

        if is_excitatory[pre_idx]:
            potentiation = np.sum(A_plus * np.exp(-delta_t_matrix[delta_t_matrix > 0] / tau_plus))
            depression = np.sum(-A_minus * np.exp(delta_t_matrix[delta_t_matrix < 0] / tau_minus))
            plasticity_impulse[post_idx, pre_idx] = potentiation + depression
        else:
            potentiation = np.sum(-A_plus * np.exp(-delta_t_matrix[delta_t_matrix > 0] / tau_plus))
            depression = np.sum(A_minus * np.exp(delta_t_matrix[delta_t_matrix < 0] / tau_minus))
            plasticity_impulse[post_idx, pre_idx] = potentiation + depression

    # --- Apply Full Update Rule ---
    # 1. Calculate the Hebbian (learning) update term
    eta_effective = eta * (1 + mod_factor)
    hebbian_update = eta_effective * plasticity_impulse

    # 2. Calculate the Weight Decay term
    W_dense = W.toarray()
    decay_term = lambda_decay * W_dense

    # 3. Apply the full update rule
    W_dense += hebbian_update - decay_term
    
    # 4. Enforce strict E/I constraints
    inhib_indices = np.where(is_excitatory == False)[0]
    excit_indices = np.where(is_excitatory == True)[0]
    
    if len(inhib_indices) > 0:
        W_dense[inhib_indices, :] = np.minimum(W_dense[inhib_indices, :], 0)
    if len(excit_indices) > 0:
        W_dense[excit_indices, :] = np.maximum(W_dense[excit_indices, :], 0)
        
    # 5. Final clipping and convert back to sparse
    W_dense = np.clip(W_dense, -2.0, 2.0)
    np.fill_diagonal(W_dense, 0)
    
    W_new = csc_matrix(W_dense)
    W_new.prune()
    
    # --- Metrics for logging ---
    net_change = W_new.toarray() - W.toarray()
    metrics = {
        'net_weight_change': np.sum(net_change),
        'potentiated_synapses': np.sum(net_change > 1e-9),
        'depressed_synapses': np.sum(net_change < -1e-9)
    }

    return W_new, metrics