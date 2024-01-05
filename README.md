# Theorectical Background
## 1) Bayes' Theorem [3]
$$P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}$$
- $P(A \vert B)$ is a conditional probability or posterior probability of $A$ given $B$.
- $P(A)$ and $P(B)$ are known as the prior probability and marginal probability.
$$P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}, \text{ if } P(B) \neq 0$$
## 2) ELBO (Evidence Lower BOund)
$$\int q_{\phi}(z \vert x)dz = 1$$
$$\begin{align}
\ln(P(x))
&= \int \ln(P(x))q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(z, x)}{P(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\frac{q_{\phi}(z \vert x)}{P(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz + \int \ln \bigg(\frac{q_{\phi}(z \vert x)}{P(z \vert x)}\bigg)q_{\phi}(z \vert x)dz
\end{align}$$
- A basic result in variational inference is that minimizing the KL-divergence is equivalent to maximizing the log-likelihood [2].
$$\begin{align}
\text{ELBO}
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(x \vert z)P(z)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \big(P(x \vert z)\big)q_{\phi}(z \vert x)dz + \int \ln \bigg(\frac{P(z)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
\end{align}$$

# References
- [1] [Auto Encoding Variational Bayes](https://github.com/KimRass/VAE/blob/main/papers/auto_encoding_variational_bayes.pdf)
- [2] [Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound)
- [3] [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
