- A supervised learning model that finds an optimal hyperplane that maximizes the margin between classes.  
- Data points:  
$$  
x \in \mathbb{R}^p  
$$  
- Labels:  
$$  
y \in \{-1, +1\}  
$$  
  
- A hyperplane in \(p\)-dimensional space is a \(p-1\) dimensional object:  
$$  
w \cdot x + b = 0  
$$
  
## Margin  
  
- The margin is the distance between the hyperplane and the closest data points (support vectors).  
  
$$  
\text{margin} = \frac{2}{\|w\|}  
$$  
  
- Maximizing the margin is equivalent to minimizing:  
$$  
\|w\|^2  
$$  
## Soft Margin Optimization

Real-world data is not perfectly separable, so we introduce slack:  
  
$$  
\min_{w,b} \; \frac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i(w \cdot x_i + b))  
$$  
  
- $C$: controls tradeoff between:  
	- large margin  
	- classification error  
## Dual Formulation  
  
Instead of solving for $w$, we solve for coefficients $\alpha_i$:  
  
$$  
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b  
$$  
  
- Only depends on dot products
- Enables use of kernels (that $K(x_i, x_j)$ term)
## Kernels 
  
Instead of explicitly mapping to higher dimensions:  
  
$$  
\phi(x)  
$$  
  
We compute:
  
$$  
K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)  
$$  
  ### Linear Kernel  
  
$$  
K(x_i, x_j) = x_i \cdot x_j  
$$  
  
- No transformation  
- Equivalent to standard linear classifier  
- Fast and interpretable

### Linear Kernel
$$
K(x_i, x_j) = x_i\cdot x_j
$$
- For linearly separable data
### Polynomial Kernel
  
$$  
K(x_i, x_j) = (x_i \cdot x_j + r)^d  
$$
- Produces curved decision boundaries based on degree
### RBF (Gaussian)
$$
K(x_i, x_j) = e^{(-\gamma||x-y||^2)}
$$
- For non-linear data, can make crazy shapes