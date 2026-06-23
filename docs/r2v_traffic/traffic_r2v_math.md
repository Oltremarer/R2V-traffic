# Traffic R2V Math

Let traffic replay be

\[
D = \{x_i\}_{i=1}^N,\quad
x_i = (s_i, a_i, r_i, s'_i, d_i, j_i, t_i, e_i).
\]

A detector gives a rarity score:

\[
\rho_i = f_{\mathrm{det}}(x_i).
\]

The candidate set is

\[
C = \{i : \rho_i \ge \tau_\rho\}
\]

or, for `not_rare_to_val`, a source set can begin outside the rare set and move toward an admitted valuable pattern.

The proposal model, with diffusion backend in the main configuration, gives a candidate or repair proposal:

\[
\tilde{x}_i = G_\theta(x_i).
\]

Admission is independent of the generator:

\[
m_i =
m_i^{\mathrm{support}}
\land
m_i^{\mathrm{ood/value}}
\land
m_i^{\mathrm{dynamics}}.
\]

Rarity \(\rho_i\) is used for detector/source/candidate diagnostics and for sampling ablations such as `rare_only`; it is not a final admission gate. This is essential for `not_rare_to_val`, where a source can start outside the rare set and still be admitted after value/support/dynamics checks pass.

The original R2V implementation also uses recovery-distance constraints around repaired samples. In the current traffic migration, that role is represented only through explicit repaired payload metadata and the conservative gate interface. A final paper claim about full original-style denoise/projection/recovery-distance behavior requires real diffusion repair artifacts with source/final gate metadata, not the proxy computed-gate path.

Traffic replay weights are

\[
w_i =
\begin{cases}
w_{\mathrm{admit}}, & m_i = 1,\\
w_{\mathrm{repair}}, & \text{proposal exists but is rejected and this mode keeps it},\\
w_{\mathrm{base}}, & \text{otherwise}.
\end{cases}
\]

Downstream training samples from

\[
\hat{\mu}_{R2V}(i) \propto w_i.
\]

The base learner target remains owned by the traffic RL backbone. R2V changes empirical replay distribution, sampling weights, or admitted transition set.
