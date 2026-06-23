# Paper Method Draft

We define each traffic transition as \(x_i=(s_i,a_i,r_i,s'_i,d_i,j_i,t_i,e_i)\). The detector computes a candidate score from traffic features or a diffusion score artifact. This score is a proposal signal only.

For `not_rare_to_val`, R2V starts from non-rare but potentially useful traffic transitions and proposes weighting them toward admitted valuable patterns. For `not_val_to_val`, it starts from rare or initially low-value candidates and asks whether a proposal can pass independent admission.

Admission is the conjunction of support, OOD/value, and dynamics gates in the full model. Gate ablations remove one gate at a time to test necessity. The final output is a weighted transition artifact with stable IDs, gate masks, and sample weights.

The downstream learner receives a changed empirical replay or pair distribution. The Bellman/PPO update form is not changed by R2V.
