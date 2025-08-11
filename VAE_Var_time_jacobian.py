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
from firedrake.pyplot import tripcolor
import pickle

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dims=[256, 128]):
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
            else:
                h = torch.tanh(h)
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


def vae_loss_function(recon_x, x, mu, logvar, beta=0.5, sigma=1.0):
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma**2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def generate_data(n_samples=1000):
    training_data = []
    for i in range(n_samples):
        noise_amplitude = 0.1
        random_field = Function(V)
        random_values = np.random.randn(*random_field.dat.data.shape) * noise_amplitude
        random_field.dat.data[:] = random_values
        
        perturbation = Function(V).interpolate(
            sin(np.random.uniform(1, 4) * math.pi * x) * 
            sin(np.random.uniform(1, 4) * math.pi * y) +
            cos(np.random.uniform(1, 3) * math.pi * x) * 
            cos(np.random.uniform(1, 3) * math.pi * y)
        )
        
        q_sample = Function(V)
        q_sample.assign(q_true_init + 0.1 * perturbation + 0.05 * random_field)
        
        training_data.append(q_sample.dat.data.copy())
    
    return np.array(training_data)

class EnhancedVAE4DVarOptimization:
    def __init__(self, vae, mean_data, std_data, latent_dim):
        self.vae = vae
        self.vae.eval()
        self.mean_data = mean_data
        self.std_data = std_data
        self.latent_dim = latent_dim
        self.device = next(vae.parameters()).device
        
    def function_to_tensor(self, func):
        #firedrake function to torch tensor
        data = (func.dat.data_ro.copy() - self.mean_data) / (self.std_data + 1e-8)
        return torch.FloatTensor(data).unsqueeze(0).to(self.device)
    
    def tensor_to_function(self, tensor, func_space):
        #tensor to firedrake function
        data = tensor.squeeze().cpu().detach().numpy()
        denormalized_data = data * (self.std_data + 1e-8) + self.mean_data
        func = Function(func_space)
        func.dat.data[:] = denormalized_data
        return func
    
    def encode_function(self, func):
        #firedrake function to latent vector
        tensor = self.function_to_tensor(func)
        with torch.no_grad():
            mu, logvar = self.vae.encode(tensor)
            return mu.squeeze().cpu().numpy()
    
    def decode_latent(self, z_array, func_space):
        #latent vector to firedrake function
        z_tensor = torch.FloatTensor(z_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.vae.decode(z_tensor)
            return self.tensor_to_function(decoded, func_space)

def train_vae(vae, training_data, num_epochs=200, batch_size=64, 
                      learning_rate=5e-4, patience=20):
    device = next(vae.parameters()).device
    # standardize training data
    mean_data = np.mean(training_data, axis=0)
    std_data = np.std(training_data, axis=0) + 1e-8
    normalized_data = (training_data - mean_data) / std_data
    
    # create DataLoader
    train_tensor = torch.FloatTensor(normalized_data)
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   patience=patience//2, factor=0.7)
    
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Training samples: {len(train_tensor)}")
    
    vae.train()
    best_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar, z = vae(data)

            beta = min(1.0, (epoch + 1) / 50)  # 前50个epoch逐渐增加β
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

        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl,
            'beta': beta,
            'lr': optimizer.param_groups[0]['lr']
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(vae.state_dict(), 'best_enhanced_vae.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch < 10:
            print(f'Epoch {epoch:3d}: Loss={avg_loss:.6f}, '
                  f'Recon={avg_recon:.6f}, KL={avg_kl:.6f}, '
                  f'β={beta:.3f}, LR={optimizer.param_groups[0]["lr"]:.2e}')
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    vae.load_state_dict(torch.load('best_enhanced_vae.pth'))
    vae.eval()
    
    return vae, mean_data, std_data, training_history

def vae4dvar_with_time():
    """真正的4D-VAE-Var：包含多时间点观测"""
    print("=== VAE-4DVar with Multiple Time Observations ===")
    
    # 生成训练数据并训练VAE
    print("Generating training data...")
    training_data = generate_data(n_samples=500)
    input_dim = training_data.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    latent_dim = 50
    vae = VAE(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        hidden_dims=[256, 128]
    ).to(device)
    
    vae, mean_data, std_data = train_vae(
        vae, training_data, num_epochs=150, batch_size=32
    )
    
    vae_optimizer = EnhancedVAE4DVarOptimization(vae, mean_data, std_data, latent_dim)
    z_background = vae_optimizer.encode_function(qb)

    continue_annotation()

    # 设置潜在变量
    Z_spaces = [FunctionSpace(mesh, "R", 0) for _ in range(latent_dim)]
    z_funcs = [Function(Z_space, name=f"latent_var_{i}") for i, Z_space in enumerate(Z_spaces)]
    z_background_funcs = [Function(Z_space) for Z_space in Z_spaces]

    for i, (z_func, z_bg_func) in enumerate(zip(z_funcs, z_background_funcs)):
        z_func.dat.data[0] = z_background[i]
        z_bg_func.dat.data[0] = z_background[i]

    def enhanced_vae_decode_function(z_funcs):
        """VAE解码函数"""
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
    J_background_terms = []
    for i in range(latent_dim):
        diff_squared = (z_funcs[i] - z_background_funcs[i])**2
        background_term = 0.5 * diff_squared * dx
        J_background_terms.append(background_term)
    
    J_background = J_background_terms[0]
    for term in J_background_terms[1:]:
        J_background = J_background + term
    
    J_background_assembled = assemble(J_background)
    
    # 计算雅可比项
    pause_annotation()
    try:
        initial_z = torch.FloatTensor(z_background).to(vae_optimizer.device)
        initial_jacobian_det = vae_optimizer.vae.compute_jacobian_determinant(initial_z, eps=1e-1)
        jacobian_constant = float(initial_jacobian_det.detach().cpu().numpy())
        print(f"Jacobian log-determinant: {jacobian_constant:.6f}")
    except Exception as e:
        print(f"Warning: Jacobian computation failed: {e}")
        jacobian_constant = 0.0
    continue_annotation()
    
    J_background_total = J_background_assembled + jacobian_constant

    # 4D观测项：多时间点观测
    print("Setting up 4D observation terms...")
    q_decoded = enhanced_vae_decode_function(z_funcs)
    q_series = solve_rk(q_decoded, return_series=True)
    
    # 观测时间索引（与传统4DVar相同）
    obs_times = [0, len(q_series)//3, 2*len(q_series)//3, -1]
    
    J_obs_total = 0
    for i, obs_idx in enumerate(obs_times):
        if obs_idx == -1:  # 最后一个时间步
            obs_idx = len(q_series) - 1
        
        if obs_idx < len(q_series):
            obs_term = assemble((H(q_series[obs_idx]) - y_obs[i])**2 * dx)
            J_obs_total = J_obs_total + obs_term
            print(f"  Time {i+1} (step {obs_idx}): observation cost = {float(obs_term):.6f}")
    
    # 总代价函数
    J_total = J_background_total + J_obs_total

    print(f"\n=== Cost Function Summary ===")
    print(f"Background cost: {float(J_background_total):.6f}")
    print(f"  - Regularization: {float(J_background_assembled):.6f}")
    print(f"  - Jacobian: {jacobian_constant:.6f}")
    print(f"Total observation cost: {float(J_obs_total):.6f}")
    print(f"Total cost: {float(J_total):.6f}")

    # 优化
    controls = [Control(z_func) for z_func in z_funcs]
    rf_vae = ReducedFunctional(J_total, controls)
    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    
    print("\nStarting VAE-4DVar optimization...")
    z_optimal_funcs = minimize(rf_vae, method="L-BFGS-B", 
                          options={"disp": True, "maxiter": 30}, 
                          derivative_options={'riesz_representation':'l2'})
    
    q_vae_optimal = enhanced_vae_decode_function(z_optimal_funcs)
    print("VAE-4DVar completed successfully")
    return q_vae_optimal

def traditional_4dvar():
    """传统4DVar方法（多时间点观测）"""
    print("=== Traditional 4D-Var ===")
    continue_annotation()
    q0 = qb.copy(deepcopy=True)
    q0.rename("q0")
    
    alpha = Constant(1e-3)
    J = assemble(alpha * (q0 - qb)**2 * dx)
    
    # 多时间点观测
    q_series = solve_rk(q0, return_series=True)
    obs_times = [0, len(q_series)//3, 2*len(q_series)//3, -1]
    
    for i, obs_idx in enumerate(obs_times):
        if obs_idx == -1:
            obs_idx = len(q_series) - 1
        if obs_idx < len(q_series):
            J = J + assemble((H(q_series[obs_idx]) - y_obs[i])**2 * dx)
    
    rf = ReducedFunctional(J, Control(q0))
    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    
    print("Starting Traditional 4D-Var optimization...")
    opt_q0_traditional = minimize(rf, method="L-BFGS-B", 
                                    options={"disp": True, "maxiter": 20}, 
                                    derivative_options={'riesz_representation':'l2'})
    print("Traditional 4D-Var completed")
    return opt_q0_traditional

def plot_4dvar_comparison(q_true_init, qb, opt_q0_traditional, q_vae_optimal, observation_points):
    """可视化4DVar结果比较"""
    obs_x = [pt[0] for pt in observation_points]
    obs_y = [pt[1] for pt in observation_points]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('4D VAE-Var vs Traditional 4D-Var\n(Multiple Time Observations with Jacobian)', 
                 fontsize=16, fontweight='bold')
    
    # 真实初始条件
    ax1 = axes[0,0]
    tripcolor(q_true_init, axes=ax1, cmap='RdBu_r', shading='gouraud')
    ax1.set_title('True Initial Condition', fontweight='bold')
    ax1.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax1.set_aspect('equal')
    
    # 背景场
    ax2 = axes[0,1]
    tripcolor(qb, axes=ax2, cmap='RdBu_r', shading='gouraud')
    ax2.set_title('Background Field', fontweight='bold')
    ax2.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax2.set_aspect('equal')
    
    # 传统4DVar结果
    ax3 = axes[1,0]
    tripcolor(opt_q0_traditional, axes=ax3, cmap='RdBu_r', shading='gouraud')
    ax3.set_title('Traditional 4D-Var', fontweight='bold')
    ax3.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax3.set_aspect('equal')
    
    # VAE-4DVar结果
    ax4 = axes[1,1]
    tripcolor(q_vae_optimal, axes=ax4, cmap='RdBu_r', shading='gouraud')
    ax4.set_title('VAE-4DVar (with Jacobian)', fontweight='bold')
    ax4.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax4.set_aspect('equal')
    
    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig("vae_4dvar_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 问题设置（从你的代码复制）
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)
    
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((0.5 - y, x - 0.5))
    u = Function(W).interpolate(velocity)
    
    # 初始条件设置
    bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
    bell = 0.25 * (1 + cos(math.pi * min_value(sqrt(pow(x - bell_x0, 2) + pow(y - bell_y0, 2)) / bell_r0, 1.0)))
    q_true_init = Function(V, name="q_true_init").interpolate(1.0 + bell)
    
    qb = Function(V, name="qb")
    qb.assign(q_true_init + 0.1 * Function(V).interpolate(sin(2 * math.pi * x) * sin(2 * math.pi * y)))
    
    # 观测配置
    observation_configs = {
        "sparse": [[0.5, 0.5], [0.25, 0.25], [0.75, 0.75]],
        "medium": [[0.25, 0.5], [0.5, 0.5], [0.75, 0.5], [0.5, 0.25], [0.5, 0.75]]
    }
    
    observation_points = observation_configs["sparse"]
    observation_mesh = VertexOnlyMesh(mesh, observation_points)
    vom = FunctionSpace(observation_mesh, "DG", 0)
    
    def H(x):
        return assemble(interpolate(x, vom))
    
    # 时间步进设置
    T = 2*math.pi
    dt = T / 400.0
    dtc = Constant(dt)
    nt = int(T / dt)
    output_freq = 20
    
    phi = TestFunction(V)
    dq_trial = TrialFunction(V)
    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))
    q_in = Constant(1.0)

    def F(q):
        return (q * div(phi * u) * dx
              - conditional(dot(u, n) < 0, phi * dot(u, n) * q_in, 0.0) * ds
              - conditional(dot(u, n) > 0, phi * dot(u, n) * q, 0.0) * ds
              - (phi('+') - phi('-')) * (un('+') * q('+') - un('-') * q('-')) * dS)

    a = phi * dq_trial * dx

    def solve_rk(q0_init, return_series=False):
        """Runge-Kutta求解器"""
        q = q0_init.copy(deepcopy=True)
        q1, q2 = Function(V), Function(V)
        dq = Function(V)
        
        L1 = dtc * F(q)
        L2 = dtc * F(q1)
        L3 = dtc * F(q2)

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        
        solv1 = LinearVariationalSolver(LinearVariationalProblem(a, L1, dq, constant_jacobian=True), solver_parameters=params)
        solv2 = LinearVariationalSolver(LinearVariationalProblem(a, L2, dq, constant_jacobian=True), solver_parameters=params)
        solv3 = LinearVariationalSolver(LinearVariationalProblem(a, L3, dq, constant_jacobian=True), solver_parameters=params)

        qs = []
        t, step = 0.0, 0
        while t < T - 0.5 * float(dt):
            solv1.solve()
            q1.assign(q + dq)
            solv2.solve()
            q2.assign(0.75 * q + 0.25 * (q1 + dq))
            solv3.solve()
            q.assign((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dq))

            step += 1
            t += float(dt)
            if step % output_freq == 0:
                qs.append(q.copy(deepcopy=True))

        return qs if return_series else q

    # 生成观测数据
    print("Generating observation data...")
    q_obs_series = solve_rk(q_true_init, return_series=True)
    obs_times = [0, len(q_obs_series)//3, 2*len(q_obs_series)//3, -1]
    y_obs = [H(q_obs_series[i]) for i in obs_times]
    
    print(f"Generated {len(y_obs)} observations at times: {obs_times}")
    print(f"Observation series length: {len(q_obs_series)}")

    # 运行实验
    print("\n" + "="*50)
    vae_results = vae4dvar_with_time()
    
    print("\n" + "="*50)
    tra_results = traditional_4dvar()
    
    # 可视化结果
    plot_4dvar_comparison(q_true_init, qb, tra_results, vae_results, observation_points)
    
    # 计算误差
    err_vae = errornorm(q_true_init, vae_results, 'L2')
    err_tra = errornorm(q_true_init, tra_results, 'L2')
    err_bg = errornorm(q_true_init, qb, 'L2')
    
    print(f"\n=== Final Results ===")
    print(f"Background L2 error: {err_bg:.6f}")
    print(f"Traditional 4D-Var L2 error: {err_tra:.6f}")
    print(f"VAE-4DVar L2 error: {err_vae:.6f}")
    print(f"VAE-4DVar improvement: {((err_tra - err_vae) / err_tra * 100):.2f}%")
    
    # 观测点误差分析
    y_true_final = H(q_obs_series[-1])
    y_vae_final = H(solve_rk(vae_results, return_series=False))
    y_tra_final = H(solve_rk(tra_results, return_series=False))
    y_bg_final = H(solve_rk(qb, return_series=False))
    
    obs_err_bg = errornorm(y_bg_final, y_true_final)
    obs_err_tra = errornorm(y_tra_final, y_true_final)
    obs_err_vae = errornorm(y_vae_final, y_true_final)
    
    print(f"\nObservation Space Errors:")
    print(f"Background: {obs_err_bg:.6f}")
    print(f"Traditional: {obs_err_tra:.6f}")
    print(f"VAE-4DVar: {obs_err_vae:.6f}")
    
    print("\n4D VAE-Var experiment completed successfully!")