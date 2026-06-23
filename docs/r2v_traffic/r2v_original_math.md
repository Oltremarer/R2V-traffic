# Original R2V Math

Let an offline dataset be \(D = \{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^N\). R2V builds a flattened vector

\[
x_i = [s_i, a_i, r_i, s'_i].
\]

A detector returns a rarity score

\[
\rho_i = f_{\mathrm{det}}(x_i).
\]

Candidates are selected by thresholding or splitting scores:

\[
C_{\mathrm{rare}} = \{i : \rho_i \ge \tau_{\rho}\}.
\]

A generative repair model proposes

\[
\tilde{x}_i = G_\theta(x_i, \epsilon, t),
\]

but \(G_\theta\) is only a proposal mechanism. Admission is controlled by independent masks:

\[
m_i =
m_i^{\mathrm{support}}
\land m_i^{\mathrm{ood/value}}
\land m_i^{\mathrm{dyn}}.
\]

The enhanced replay distribution is represented by weights

\[
w_i =
\begin{cases}
w_{\mathrm{admit}}, & m_i = 1 \\
w_{\mathrm{base}}, & m_i = 0
\end{cases}
\]

or by appending repaired/admitted samples when that mode is enabled. Downstream RL samples from the altered empirical replay distribution

\[
\hat{\mu}_{R2V}(i) \propto w_i,
\]

while keeping the learner target, such as

\[
y = r + \gamma(1-d) Q_{\bar{\theta}}(s', a'),
\]

owned by the backbone algorithm.
