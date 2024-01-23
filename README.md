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
- Mean and STD of MNIST Test Set
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/d375b8c2-71ae-488a-b46e-7e1c8897ba9c" width="600">
    - 평균의 경우 4와 9, 3과 5가 많이 겹쳐 있습니다.
    - 표준편차의 경우 1에 가까워지도록 학습이 이루어졌으나 0에 가까운 값을 띄고 있습니다. 시각화를 통해 얻을 수 있는 인사이트는 크게 없는 것으로 보입니다.
## 2) Decoder Output
```bash
python3 vis/decoder_output/main.py\
    --seed=888\ # Optional
    --latent_min=-4\ # Optional
    --latent_max=-4\ # Optional
    --n_cells=32\ # Optional
    --model_params="/.../datasets/vae/vae_mnist.pth"\
    --data_dir="/.../datasets"\
    --save_dir="/.../workspace/VAE/vis/encoder_output"
```
- `latent_min`: -4, `latent_max`: 4, `n_cells`: 32
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/d14e782f-9e9b-4bd3-b04c-2bdf84a032ce" width="600">
    - Encoder output의 평균의 분포와 매우 유사함을 볼 수 있습니다.
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
- A basic result in variational inference is that latent_minimizing the KL-divergence is equivalent to latent_maximizing the log-likelihood [2].
$$
\begin{align}
\text{ELBO}
&= \int \ln \bigg(\frac{P(z, x)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \bigg(\frac{P(x \vert z)P(z)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
&= \int \ln \big(P(x \vert z)\big)q_{\phi}(z \vert x)dz + \int \ln \bigg(\frac{P(z)}{q_{\phi}(z \vert x)}\bigg)q_{\phi}(z \vert x)dz\\
\end{align}
$$
<!-- ## 왜 reparametrization trick?
## KLD loss 공식 유도
## 논문 읽기, Monte Carlo, 왜 샘플링 한 번만? -->

# 4. References
- [1] [Auto Encoding Variational Bayes](https://github.com/KimRass/VAE/blob/main/papers/auto_encoding_variational_bayes.pdf)
- [2] [Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound)
- [3] [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
