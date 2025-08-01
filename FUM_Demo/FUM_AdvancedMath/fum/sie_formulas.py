"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

# FUM_AdvancedMath.fum.sie_formulas
#
# Provides the pure, canonical implementations of the mathematical formulas
# for the Self-Improvement Engine (SIE).
# 
# Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
#
# This research is protected under a dual-license to foster open academic
# research while ensuring commercial applications are aligned with the project's 
# ethical principles. 
# Commercial use requires written permission from Justin K. Lietz. 
# See LICENSE file for full terms.
# ===========================================================================

import torch

def calculate_td_error(V_current: float, R_external: float, V_next: float, gamma: float) -> float:
    """
    Calculates the Temporal Difference (TD) error for a state transition.
    Ref: Blueprint Rule 3 (Component of the SIE)
    Time Complexity: O(1)
    """
    return R_external + (gamma * V_next) - V_current

def calculate_novelty_score(N_s: int) -> float:
    """
    Calculates the novelty score for a state based on its visitation count.
    Ref: Blueprint Rule 3 (Component of the SIE)
    Time Complexity: O(1)
    """
    # Inverse visitation count, add epsilon for stability
    return 1.0 / (N_s + 1e-6)

def calculate_habituation_score(encoding_current: torch.Tensor, encoding_history: list) -> float:
    """
    Calculates a habituation score based on the similarity of the current
    input encoding to a recent history of encodings.
    Ref: Blueprint Rule 3 (Component of the SIE)
    Time Complexity: O(L*d) where L is history length, d is encoding dim.
    """
    if not encoding_history:
        return 0.0
    
    history_tensor = torch.stack(encoding_history)
    # Using cosine similarity to measure similarity
    similarities = torch.nn.functional.cosine_similarity(encoding_current, history_tensor, dim=1)
    return torch.mean(similarities).item()

def calculate_hsi(firing_rates: torch.Tensor, target_var: float) -> float:
    """
    Calculates the Homeostatic Stability Index (HSI).
    Ref: Blueprint Rule 3.1 & FUM Nomenclature
    Time Complexity: O(N) where N is number of neurons.
    """
    current_var = torch.var(firing_rates)
    return 1.0 - (torch.abs(current_var - target_var) / target_var)

def calculate_total_reward(w_td: float, td_error_norm: float,
                           w_nov: float, novelty_norm: float,
                           w_hab: float, habituation_norm: float,
                           w_hsi: float, hsi_norm: float) -> float:
    """
    Calculates the composite total_reward signal from its four weighted,
    normalized components.
    Ref: Blueprint Rule 3
    Time Complexity: O(1)
    """
    reward = (w_td * td_error_norm +
              w_nov * novelty_norm -
              w_hab * habituation_norm +
              w_hsi * hsi_norm)
    return reward