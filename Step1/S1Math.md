# Tree Crown Model - Mathematical Formulation

## 1. Spheroid Shape (Soft Membership)

A point $\mathbf{p} = (x, y, z)$ belongs to a **prolate spheroid** centered at $\boldsymbol{\mu} = (\mu_x, \mu_y, \mu_z)$ with:
- Horizontal radius: $r_{xy}$ (same for x and y directions)
- Vertical radius: $r_z$

The normalized squared distance is:

$$
d^2(\mathbf{p}) = \left(\frac{x - \mu_x}{r_{xy}}\right)^2 + 
\left(\frac{y - \mu_y}{r_{xy}}\right)^2 + 
\left(\frac{z - \mu_z}{r_z}\right)^2
$$

We use a **soft sigmoid boundary** instead of hard cutoff:

$$
w(\mathbf{p}) = \sigma\Big(4 \cdot \big(1 - d^2(\mathbf{p})\big)\Big)
$$

Where $\sigma$ is the sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Python Implementation:**
```python
def inside_weight(self, points):
    dx = (x - μ_x) / r_xy
    dy = (y - μ_y) / r_xy  
    dz = (z - μ_z) / r_z
    dist_sq = dx² + dy² + dz²
    return torch.sigmoid(4.0 * (1.0 - dist_sq))
```

## 2. LiDAR Physics (Beer-Lambert Law)

The crown top height is:

$$
z_{\text{top}} = \mu_z + r_z
$$

Vertical path length from point to crown top:

$$
\Delta z(\mathbf{p}) = \max\big(0,\; z_{\text{top}} - z\big)
$$

Optical depth (attenuation) with foliage density $\lambda$:

$$
\tau(\mathbf{p}) = \lambda \cdot w(\mathbf{p}) \cdot \Delta z(\mathbf{p})
$$

## 3. Probability Density Function

The likelihood of observing a LiDAR point $\mathbf{p}$:

$$
P(\mathbf{p}) = \underbrace{\lambda \cdot w(\mathbf{p})}_{\text{density at point}} \cdot 
\underbrace{\exp\!\big(-\tau(\mathbf{p})\big)}_{\text{survival probability}}
$$

**Full expanded form:**

$$
P(x,y,z) = \lambda \cdot \sigma\Big(4\big[1 - (\frac{x-\mu_x}{r_{xy}})^2 - (\frac{y-\mu_y}{r_{xy}})^2 - (\frac{z-\mu_z}{r_z})^2\big]\Big) \cdot
\exp\!\Big(-\lambda \cdot \sigma(\cdots) \cdot \max(0, \mu_z + r_z - z)\Big)
$$

## 4. Optimization Loss

We maximize log-likelihood with physics constraints:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log P(\mathbf{p}_i) + \text{penalties}
$$

**Penalty terms:**
1. Height constraint: $5 \cdot \max(0, z_{p95} - z_{\text{top}})^2$
2. Lambda regularization: $0.001 \cdot (\log \lambda)^2$
3. Minimum radii: $2 \cdot \max(0, 0.5 - r_{xy})^2 + 2 \cdot \max(0, 0.8 - r_z)^2$

---

## Quick Test Your Setup

Type this simple equation in your `.md` file:
```latex
$$
E = mc^2
$$
```

Then press `Ctrl+Shift+V` to preview. You should see: 

$$
E = mc^2
$$

If you see the rendered equation, your Markdown+Math is working!