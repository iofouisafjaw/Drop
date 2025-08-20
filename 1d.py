import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate 
from firedrake.adjoint import *
from pyadjoint import ReducedFunctional, Control, minimize
import math
import matplotlib.pyplot as plt
from pyadjoint.tape import pause_annotation, continue_annotation
import pickle

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dims=[128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # encoder
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        prev_dim = hidden_dims[-1]
        for hidden_dim in reversed(hidden_dims[:-1]):
            self.decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = F.gelu(layer(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i < len(self.decoder_layers) - 1:
                h = F.gelu(h)
        return h
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def compute_decoder_jacobian(self, z):
        if z.dim() == 1:
            z_expanded = z.unsqueeze(0)
        else:
            z_expanded = z
        
        z_expanded = z_expanded.clone().detach().requires_grad_(True)
        decoded = self.decode(z_expanded)
        
        batch_size, output_dim = decoded.shape
        latent_dim = z_expanded.shape[1]
        
        jacobian = torch.zeros(output_dim, latent_dim, device=z.device)
        
        for i in range(output_dim):
            grad_outputs = torch.zeros_like(decoded)
            grad_outputs[0, i] = 1.0

            grads = torch.autograd.grad(
                    outputs=decoded,
                    inputs=z_expanded,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False
                )[0]
            jacobian[i] = grads[0]
        return jacobian
    
    def compute_jacobian_determinant(self, z, eps=1e-2):
        jacobian = self.compute_decoder_jacobian(z)
        jtj = torch.matmul(jacobian.t(), jacobian)
        jtj_reg = jtj + eps * torch.eye(jtj.shape[0], device=jtj.device)
        det_jtj = torch.det(jtj_reg)
        det_jtj = torch.clamp(det_jtj, min=1e-10)
        log_det = 0.5 * torch.log(det_jtj)
        return log_det

def vae_loss_function(recon_x, x, mu, logvar, beta, sigma):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma**2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def generate_1d_data(V, x, q_true0, n_samples=1000):
    """生成1D训练数据"""
    print(f"Generating {n_samples} 1D training samples...")
    training_data = []
    
    for i in range(n_samples):
        # 添加随机噪声
        noise_amplitude = 0.05
        random_field = Function(V)
        random_values = np.random.randn(*random_field.dat.data.shape) * noise_amplitude
        random_field.dat.data[:] = random_values
        
        # 添加随机正弦波扰动
        freq1 = np.random.uniform(1, 5)
        freq2 = np.random.uniform(1, 5)
        phase1 = np.random.uniform(0, 2*math.pi)
        phase2 = np.random.uniform(0, 2*math.pi)
        amp1 = np.random.uniform(0.05, 0.15)
        amp2 = np.random.uniform(0.05, 0.15)
        
        perturbation = Function(V).interpolate(
            amp1 * sin(freq1 * math.pi * x + phase1) +
            amp2 * cos(freq2 * math.pi * x + phase2)
        )
        
        # 创建样本
        q_sample = Function(V)
        q_sample.assign(q_true0 + perturbation + random_field)
        
        training_data.append(q_sample.dat.data.copy())
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")
    
    return np.array(training_data)

class VAE4DVarOptimization:
    def __init__(self, vae, mean_data, std_data, latent_dim):
        self.vae = vae
        self.vae.eval()
        self.mean_data = mean_data
        self.std_data = std_data
        self.latent_dim = latent_dim
        self.device = next(vae.parameters()).device
        
    def function_to_tensor(self, func):
        """Firedrake函数转换为PyTorch张量"""
        data = (func.dat.data_ro.copy() - self.mean_data) / (self.std_data + 1e-8)
        return torch.FloatTensor(data).unsqueeze(0).to(self.device)
    
    def tensor_to_function(self, tensor, func_space):
        """PyTorch张量转换为Firedrake函数"""
        data = tensor.squeeze().cpu().detach().numpy()
        denormalized_data = data * (self.std_data + 1e-8) + self.mean_data
        func = Function(func_space)
        func.dat.data[:] = denormalized_data
        return func
    
    def encode_function(self, func):
        """编码Firedrake函数到潜在空间"""
        tensor = self.function_to_tensor(func)
        with torch.no_grad():
            mu, logvar = self.vae.encode(tensor)
            return mu.squeeze().cpu().numpy()
    
    def decode_latent(self, z_array, func_space):
        """从潜在空间解码到Firedrake函数"""
        z_tensor = torch.FloatTensor(z_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.vae.decode(z_tensor)
            return self.tensor_to_function(decoded, func_space)

def train_vae(vae, training_data, num_epochs=150, batch_size=32, learning_rate=5e-4):
    """训练VAE"""
    print("Training VAE...")
    device = next(vae.parameters()).device
    
    # 标准化训练数据
    mean_data = np.mean(training_data, axis=0)
    std_data = np.std(training_data, axis=0) + 1e-8
    normalized_data = (training_data - mean_data) / std_data
    
    # 创建数据加载器
    train_tensor = torch.FloatTensor(normalized_data)
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.7)
    
    print(f"Device: {device}")
    print(f"Training samples: {len(train_tensor)}")
    print(f"Input dimension: {training_data.shape[1]}")
    print(f"Latent dimension: {vae.latent_dim}")
    
    vae.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar, z = vae(data)

            # 逐渐增加KL散度权重
            beta = min(1.0, (epoch + 1) / 50)
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, data, mu, logvar, beta=beta, sigma=1.0
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon_loss / len(train_loader.dataset)
        avg_kl = total_kl_loss / len(train_loader.dataset)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(vae.state_dict(), 'best_vae_1d.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Loss={avg_loss:.6f}, '
                  f'Recon={avg_recon:.6f}, KL={avg_kl:.6f}, '
                  f'β={beta:.3f}, LR={optimizer.param_groups[0]["lr"]:.2e}')
            
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break
    
    vae.load_state_dict(torch.load('best_vae_1d.pth'))
    vae.eval()
    
    return vae, mean_data, std_data

def vae_4dvar_1d(mesh, V, q_true0, qb, y_obs_series, H, forward_solve):
    # 生成训练数据
    x = SpatialCoordinate(mesh)[0]
    training_data = generate_1d_data(V, x, q_true0, n_samples=800)
    input_dim = training_data.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 构建并训练VAE
    latent_dim = 15  # 1D情况下使用较小的潜在维度
    vae = VAE(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        hidden_dims=[128, 64]
    ).to(device)
    
    vae, mean_data, std_data = train_vae(vae, training_data, num_epochs=150, batch_size=32)
    
    # 创建VAE优化器
    vae_optimizer = VAE4DVarOptimization(vae, mean_data, std_data, latent_dim)
    z_background = vae_optimizer.encode_function(qb)
    
    print(f"Compression ratio: {input_dim / latent_dim:.1f}:1")
    print(f"Background encoded to latent space: {z_background.shape}")

    # 开始4D-Var优化
    continue_annotation()
    z_spaces = [FunctionSpace(mesh, "R", 0) for _ in range(latent_dim)]
    z_funcs = [Function(Z_space, name=f"latent_var_{i}") for i, Z_space in enumerate(z_spaces)]
    z_background_funcs = [Function(Z_space) for Z_space in z_spaces]
    for i, (z_func, z_bg_func) in enumerate(zip(z_funcs, z_background_funcs)):
        z_func.dat.data[0] = z_background[i]
        z_bg_func.dat.data[0] = z_background[i]

    def vae_decode_function(z_funcs):
        z_array = np.array([z_func.dat.data[0] for z_func in z_funcs])
        pause_annotation()
        
        z_tensor = torch.FloatTensor(z_array).unsqueeze(0).to(vae_optimizer.device)
        with torch.no_grad():
            decoded_tensor = vae_optimizer.vae.decode(z_tensor)
            decoded_data = decoded_tensor.squeeze().cpu().detach().numpy()
        denormalized_data = decoded_data * (vae_optimizer.std_data + 1e-8) + vae_optimizer.mean_data

        continue_annotation()
        q_decoded = Function(V, name="decoded_field")
        q_decoded.dat.data[:] = denormalized_data
        return q_decoded

    # 背景项
    alpha_vae = 1e-3 
    J_background_terms = []
    for i in range(latent_dim):
        diff_squared = (z_funcs[i] - z_background_funcs[i])**2
        background_term = 0.5 * alpha_vae * diff_squared * dx
        J_background_terms.append(background_term)
    
    J_background = J_background_terms[0]
    for term in J_background_terms[1:]:
        J_background = J_background + term
    J_background_assembled = assemble(J_background)
    
    # 雅可比行列式项（用于体积校正）
    pause_annotation()
    initial_z = torch.FloatTensor(z_background).to(vae_optimizer.device)
    initial_jacobian_det = vae_optimizer.vae.compute_jacobian_determinant(initial_z, eps=1e-2)
    jacobian_constant = float(initial_jacobian_det.detach().cpu().numpy())
    print(f"Initial Jacobian log-determinant: {jacobian_constant}")
    continue_annotation()

    J_background_total = J_background_assembled + 0.1 * jacobian_constant

    q_decoded = vae_decode_function(z_funcs)
    q_series = forward_solve(q_decoded)
    
    J_obs = 0.0
    sigma2 = 1e-4
    for qk, yk in zip(q_series, y_obs_series):
        diff = H(qk) - yk
        J_obs += 0.5 * (1.0/sigma2) * assemble(diff*diff * dx)

    J_total = J_background_total + J_obs
    controls = [Control(z_func) for z_func in z_funcs]
    rf_vae = ReducedFunctional(J_total, controls)
    
    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    
    print("Starting VAE-4DVar optimization...")
    z_optimal_funcs = minimize(rf_vae, method="L-BFGS-B", 
                              options={"disp": True, "maxiter": 30}, 
                              derivative_options={'riesz_representation':'l2'})
    
    q_vae_optimal = vae_decode_function(z_optimal_funcs) 
    return q_vae_optimal, vae_optimizer

def traditional_4dvar_1d(qb, y_obs_series, H, forward_solve):
    print("\n" + "="*50)
    print("Starting Traditional 4D-Var optimization...")
    print("="*50)
    
    continue_annotation()
    q0 = qb.copy(deepcopy=True)
    q0.rename("q0_traditional")

    alpha_trad = 1e-2
    J_bg = 0.5 * alpha_trad * assemble((q0 - qb)**2 * dx)
    q_series = forward_solve(q0)

    J_obs = 0.0
    sigma2 = 1e-4
    for qk, yk in zip(q_series, y_obs_series):
        diff = H(qk) - yk
        J_obs += 0.5 * (1.0/sigma2) * assemble(diff*diff * dx)
    
    J_total = J_bg + J_obs
    rf = ReducedFunctional(J_total, Control(q0))

    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    opt_q0_traditional = minimize(rf, method="L-BFGS-B", 
                                 options={"disp": True, "maxiter": 30}, 
                                 derivative_options={'riesz_representation':'l2'})

    return opt_q0_traditional

#visualize
def plot_1d_comparison(q_true0, qb, opt_q0_traditional, q_vae_optimal, obs_points, mesh):
    data_length = len(q_true0.dat.data)
    x_plot = np.linspace(0, 1, data_length)
    obs_x = [pt[0] for pt in obs_points]

    q_true_data = q_true0.dat.data
    qb_data = qb.dat.data
    trad_data = opt_q0_traditional.dat.data
    vae_data = q_vae_optimal.dat.data

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1D VAE-4DVar vs Traditional 4D-Var Comparison', fontsize=16, fontweight='bold')

    axes[0,0].plot(x_plot, q_true_data, 'k-', linewidth=2, label='Truth')
    axes[0,0].plot(x_plot, qb_data, 'b--', linewidth=2, label='Background')
    axes[0,0].plot(x_plot, trad_data, 'g-', linewidth=2, label='Traditional 4D-Var')
    axes[0,0].plot(x_plot, vae_data, 'r-', linewidth=2, label='VAE-4DVar')
    
    for obs_pt in obs_x:
        axes[0,0].axvline(x=obs_pt, color='orange', linestyle=':', alpha=0.7)
    
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('q(x,0)')
    axes[0,0].set_title('Initial Conditions Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    error_bg = qb_data - q_true_data
    error_trad = trad_data - q_true_data
    error_vae = vae_data - q_true_data
    
    axes[0,1].plot(x_plot, error_bg, 'b--', linewidth=2, label='Background Error')
    axes[0,1].plot(x_plot, error_trad, 'g-', linewidth=2, label='Traditional Error')
    axes[0,1].plot(x_plot, error_vae, 'r-', linewidth=2, label='VAE Error')
    axes[0,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('Error')
    axes[0,1].set_title('Error Analysis')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    rms_bg = np.sqrt(np.mean(error_bg**2))
    rms_trad = np.sqrt(np.mean(error_trad**2))
    rms_vae = np.sqrt(np.mean(error_vae**2))
    
    methods = ['Background', 'Traditional', 'VAE-4DVar']
    rms_values = [rms_bg, rms_trad, rms_vae]
    colors = ['blue', 'green', 'red']
    
    bars = axes[1,0].bar(methods, rms_values, color=colors, alpha=0.7)
    axes[1,0].set_ylabel('RMS Error')
    axes[1,0].set_title('RMS Error Comparison')
    axes[1,0].grid(True, alpha=0.3)

    for bar, val in zip(bars, rms_values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                      f'{val:.4f}', ha='center', va='bottom')
        
    zoom_region = slice(int(0.4*data_length), int(0.6*data_length))
    x_zoom = x_plot[zoom_region]
    
    axes[1,1].plot(x_zoom, q_true_data[zoom_region], 'k-', linewidth=2, label='Truth')
    axes[1,1].plot(x_zoom, trad_data[zoom_region], 'g-', linewidth=2, label='Traditional')
    axes[1,1].plot(x_zoom, vae_data[zoom_region], 'r-', linewidth=2, label='VAE-4DVar')
    for obs_pt in obs_x:
        if 0.4 <= obs_pt <= 0.6:
            axes[1,1].axvline(x=obs_pt, color='orange', linestyle=':', alpha=0.7)
    
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('q(x,0)')
    axes[1,1].set_title('Zoomed View (0.4 ≤ x ≤ 0.6)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("vae_4dvar_1d_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Background RMS error:      {rms_bg:.6f}")
    print(f"Traditional 4D-Var error:  {rms_trad:.6f}")
    print(f"VAE-4DVar error:           {rms_vae:.6f}")
    print(f"Traditional improvement:   {((rms_bg-rms_trad)/rms_bg)*100:.2f}%")
    print(f"VAE improvement:           {((rms_bg-rms_vae)/rms_bg)*100:.2f}%")

def main():
    L = 1.0
    N = 100
    p = 1
    aval = 1.0
    T = 1.0
    nt = 300
    output_freq = 20
    
    # obs points
    obs_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    obs_points = [[xi] for xi in obs_points]

    mesh = PeriodicIntervalMesh(N, L)
    V = FunctionSpace(mesh, "DG", p)
    x = SpatialCoordinate(mesh)[0]
    a = Constant(aval)
    
    # q_true
    wave1 = 0.3 * sin(2 * math.pi * x)           # 基础波
    wave2 = 0.2 * sin(4 * math.pi * x + 0.5)     # 高频波，带相位
    wave3 = 0.15 * sin(6 * math.pi * x - 0.3)    # 更高频波
    wave4 = 0.1 * cos(3 * math.pi * x + 1.2)     # 余弦波添加复杂性
    
    # 组合成复杂的初始条件
    q_true0 = Function(V, name="q_true0").interpolate(
        1.0 + wave1 + wave2 + wave3 + wave4
    )
    
    # qb
    bg_wave1 = 0.25 * sin(2 * math.pi * x + 0.2)  
    bg_wave2 = 0.18 * sin(4 * math.pi * x + 0.8)  
    bg_wave3 = 0.12 * cos(5 * math.pi * x - 0.1)  
    
    qb = Function(V, name="qb").interpolate(
        1.0 + bg_wave1 + bg_wave2 + bg_wave3
    ) 

    obs_mesh = VertexOnlyMesh(mesh, obs_points)
    Vom = FunctionSpace(obs_mesh, "DG", 0)
    def H(x):
        return assemble(interpolate(x, Vom))

    phi = TestFunction(V)
    dq_trial = TrialFunction(V)
    n = FacetNormal(mesh)
    u_up = lambda w: conditional(gt(a*n('-')[0], 0), w('-'), w('+'))
    M_form = phi * dq_trial * dx
    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
    def forward_solve(q0):
        q = Function(V, name="q")
        q.assign(q0)
        q1, q2 = Function(V), Function(V)
        dq = Function(V)
        
        dt = Constant(T/nt)
        
        def RHS(qcur):
            return - a * (qcur * phi.dx(0)) * dx + a * (u_up(qcur) * jump(phi)) * dS
        
        series = []
        t = 0.0
        step = 0
        
        while t < float(T) - 0.5*float(T/nt):
            # RK3 第一步
            L1 = dt * RHS(q)
            solve(M_form == L1, dq, solver_parameters=params)
            q1.assign(q + dq)
            
            # RK3 第二步
            L2 = dt * RHS(q1)
            solve(M_form == L2, dq, solver_parameters=params)
            q2.assign(0.75*q + 0.25*(q1 + dq))
            
            # RK3 第三步
            L3 = dt * RHS(q2)
            solve(M_form == L3, dq, solver_parameters=params)
            q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
            
            step += 1
            t += float(T/nt)
            
            if step % output_freq == 0:
                series.append(q.copy(deepcopy=True, annotate=True))
        
        return series
    print("Generating observations...")
    q_true_series = forward_solve(q_true0)
    y_obs_series = [H(qk) for qk in q_true_series]
    for y in y_obs_series:
        noise = 0.0 * np.random.normal(0, 0.01, y.dat.data.shape)  
        y.dat.data[:] += noise
    
    print(f"Generated {len(y_obs_series)} observation time steps")
    q_vae_optimal, vae_optimizer = vae_4dvar_1d(mesh, V, q_true0, qb, y_obs_series, H, forward_solve)
    opt_q0_traditional = traditional_4dvar_1d(qb, y_obs_series, H, forward_solve)

    plot_1d_comparison(q_true0, qb, opt_q0_traditional, q_vae_optimal, obs_points, mesh)
    
    # 计算L2误差
    err_vae = errornorm(q_true0, q_vae_optimal, 'L2')
    err_trad = errornorm(q_true0, opt_q0_traditional, 'L2')
    
    print(f"\nFinal L2 Error Comparison:")
    print(f"VAE-4DVar L2 error:     {err_vae:.6f}")
    print(f"Traditional L2 error:   {err_trad:.6f}")
    if err_trad > err_vae:
        print(f"VAE improvement:        {((err_trad - err_vae) / err_trad * 100):.2f}%")
    else:
        print(f"Traditional better by:  {((err_vae - err_trad) / err_vae * 100):.2f}%")
    
    # 保存结果
    print("\nSaving results...")
    results = {
        'q_true': q_true0.dat.data,
        'q_background': qb.dat.data,
        'q_traditional': opt_q0_traditional.dat.data,
        'q_vae': q_vae_optimal.dat.data,
        'obs_points': obs_points,
        'err_vae': err_vae,
        'err_trad': err_trad
    }

def visualize_vae_latent_space(vae_optimizer, q_true0, qb, q_vae_optimal):
    """可视化VAE潜在空间"""
    print("\nVisualizing VAE latent space...")
    
    # 编码不同的函数到潜在空间
    z_true = vae_optimizer.encode_function(q_true0)
    z_bg = vae_optimizer.encode_function(qb)
    z_opt = vae_optimizer.encode_function(q_vae_optimal)
    
    # 可视化前几个潜在维度
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 潜在变量值
    latent_dims = min(10, len(z_true))
    x_latent = np.arange(latent_dims)
    
    axes[0].bar(x_latent, z_true[:latent_dims], alpha=0.7, label='True', color='black')
    axes[0].bar(x_latent, z_bg[:latent_dims], alpha=0.7, label='Background', color='blue')
    axes[0].bar(x_latent, z_opt[:latent_dims], alpha=0.7, label='VAE Optimal', color='red')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Latent Value')
    axes[0].set_title('Latent Space Representation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 重构误差
    q_true_recon = vae_optimizer.decode_latent(z_true, q_true0.function_space())
    q_bg_recon = vae_optimizer.decode_latent(z_bg, qb.function_space())
    
    recon_err_true = np.mean((q_true0.dat.data - q_true_recon.dat.data)**2)
    recon_err_bg = np.mean((qb.dat.data - q_bg_recon.dat.data)**2)
    
    methods = ['True', 'Background']
    recon_errors = [recon_err_true, recon_err_bg]
    
    axes[1].bar(methods, recon_errors, color=['black', 'blue'], alpha=0.7)
    axes[1].set_ylabel('Reconstruction MSE')
    axes[1].set_title('VAE Reconstruction Quality')
    axes[1].grid(True, alpha=0.3)
    
    # 潜在空间距离
    l2_dist_bg = np.linalg.norm(z_true - z_bg)
    l2_dist_opt = np.linalg.norm(z_true - z_opt)
    
    axes[2].bar(['Background', 'VAE Optimal'], [l2_dist_bg, l2_dist_opt], 
               color=['blue', 'red'], alpha=0.7)
    axes[2].set_ylabel('L2 Distance in Latent Space')
    axes[2].set_title('Distance from Truth in Latent Space')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("vae_latent_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Latent space dimension: {len(z_true)}")
    print(f"True reconstruction MSE: {recon_err_true:.6f}")
    print(f"Background reconstruction MSE: {recon_err_bg:.6f}")
    print(f"Background latent distance: {l2_dist_bg:.6f}")
    print(f"Optimal latent distance: {l2_dist_opt:.6f}")

if __name__ == "__main__":
    # 运行主程序
    main()
    print(2)