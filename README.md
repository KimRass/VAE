# 1. Pre-trained Parameters
- Trained on MNIST for 84 epochs ([vae_mnist.pth](https://drive.google.com/file/d/1RLy035sMe-Wn1bgB9A60nmCc_tZbK3t7/view?usp=sharing))
    - `seed`: 888, `recon_weight`: 600, `lr`: 0.0005, `batch_size`: 64
    - `val_recon_loss`: 0.1085, `val_kld_loss`: 7.3032

# 2. Visualization
## 1) Encoder Output
```bash
# e.g.,
python3 vis/encoder_output/main.py\
    --seed=888\ # Optional
    --batch_size=64\ # Optional
    --taget="mean"\ # Or `"std"`
    --model_params="/.../datasets/vae/vae_mnist.pth"\
    --data_dir="/.../datasets"\
    --save_dir="/.../workspace/VAE/vis/encoder_output"
```
- Mean of MNIST Test Set
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/f1c33bc9-96fa-4f01-8bf5-f182e3bb0b13" width="400">
- Standard Deviation of MNIST Test Set
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/fa0a9e74-8737-44d4-b731-34bc6b1422a2" width="400">
## 2) Decoder Output
```bash
python3 vis/decoder_output/main.py\
    --seed=888\ # Optional
    --n_cells=20\ # Optional
    --model_params="/.../datasets/vae/vae_mnist.pth"\
    --data_dir="/.../datasets"\
    --save_dir="/.../workspace/VAE/vis/encoder_output"
```
- Value from -2 to 2
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/3febd7b3-9e8f-43db-ad16-af616a3428c3" width="400">
## 3) Image Reconstruction
```bash
python3 vis/reconstruct/main.py\
    --seed=888\ # Optional
    --batch_size=128\ # Optional
    --model_params="/.../datasets/vae/vae_mnist.pth"\
    --data_dir="/.../datasets"\
    --save_dir="/.../workspace/VAE/vis/encoder_output"
```
- MNIST Test Set
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/c08037a5-de9e-411a-81f4-6921b07fd402" width="800">

# 3. Theoretical Background
## 1) Bayes' Theorem [3]
$$P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}$$
- $P(A \vert B)$ is a conditional probability or posterior probability of $A$ given $B$.
- $P(A)$ and $P(B)$ are known as the prior probability and marginal probability.
$$P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}, \text{ if } P(B) \neq 0$$
## 2) ELBO (Evidence Lower BOund)
$$\int q_{\phi}(z \vert x)dz = 1$$
$$
\begin{align}
\ln(P(x))
&= \int \ln(P(x))q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(z, x)}{P(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\frac{q_{\phi}(z \vert x)}{P(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz + \int \ln \bigg(\frac{q_{\phi}(z \vert x)}{P(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
\end{align}
$$
- A basic result in variational inference is that minimizing the KL-divergence is equivalent to maximizing the log-likelihood [2].
$$
\begin{align}
\text{ELBO}
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(x \vert z)P(z)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \big(P(x \vert z)\big)q_{\phi}(z \vert x)dz + \int \ln \bigg(\frac{P(z)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
\end{align}
$$

# 4. References
- [1] [Auto Encoding Variational Bayes](https://github.com/KimRass/VAE/blob/main/papers/auto_encoding_variational_bayes.pdf)
- [2] [Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound)
- [3] [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
