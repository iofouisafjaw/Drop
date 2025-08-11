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
    def __init__(self, input_dim, latent_dim=20, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  
                nn.GELU(), 
                nn.Dropout(0.1) 
            ])
            prev_dim = hidden_dim
            print(f"   Encoder layer {i+1}: {prev_dim} -> {hidden_dim}")
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # latent variables
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # decoder
        decoder_layers = []
        decoder_layers.extend([
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(0.1)
        ])
        
        prev_dim = hidden_dims[-1]
        for i, hidden_dim in enumerate(reversed(hidden_dims[:-1])):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            print(f"   Decoder layer {i+1}: {prev_dim} -> {hidden_dim}")
        
        # output layer
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Tanh()  
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, beta=0.5, sigma=1.0):
    batch_size = x.size(0)
    
    # weighted mse
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma**2)
    
    # KL loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def generate_data(n_samples=1000): #just using simple way to generate data
    training_data = []
    for i in range(n_samples):
        # Create random perturbations
        noise_amplitude = 0.1
        random_field = Function(V)
        random_values = np.random.randn(*random_field.dat.data.shape) * noise_amplitude
        random_field.dat.data[:] = random_values
        
        # Create varied initial conditions
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
        """Convert Firedrake function to normalized tensor"""
        data = (func.dat.data_ro.copy() - self.mean_data) / (self.std_data + 1e-8)
        return torch.FloatTensor(data).unsqueeze(0).to(self.device)
    
    def tensor_to_function(self, tensor, func_space):
        """Convert tensor back to Firedrake function"""
        data = tensor.squeeze().cpu().detach().numpy()
        denormalized_data = data * (self.std_data + 1e-8) + self.mean_data
        func = Function(func_space)
        func.dat.data[:] = denormalized_data
        return func
    
    def encode_function(self, func):
        """Encode Firedrake function to latent space"""
        tensor = self.function_to_tensor(func)
        with torch.no_grad():
            mu, logvar = self.vae.encode(tensor)
            return mu.squeeze().cpu().numpy()
    
    def decode_latent(self, z_array, func_space):
        """Decode latent vector to Firedrake function"""
        z_tensor = torch.FloatTensor(z_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.vae.decode(z_tensor)
            return self.tensor_to_function(decoded, func_space)

def train_enhanced_vae(vae, training_data, num_epochs=200, batch_size=64, 
                      learning_rate=5e-4, patience=20):
    device = next(vae.parameters()).device
    
    # 数据标准化
    mean_data = np.mean(training_data, axis=0)
    std_data = np.std(training_data, axis=0) + 1e-8
    normalized_data = (training_data - mean_data) / std_data
    
    # 创建数据加载器
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
            
            # forward probagation
            recon_batch, mu, logvar, z = vae(data)
            
            # 计算损失 - β退火策略
            beta = min(1.0, (epoch + 1) / 50)  # 前50个epoch逐渐增加β
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, data, mu, logvar, beta=beta, sigma=1.0
            )
            
            # backward probagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon_loss / len(train_loader.dataset)
        avg_kl = total_kl_loss / len(train_loader.dataset)
        
        # learning rate scheduler
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
            # save best model
            torch.save(vae.state_dict(), 'best_enhanced_vae.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch < 10:
            print(f'Epoch {epoch:3d}: Loss={avg_loss:.6f}, '
                  f'Recon={avg_recon:.6f}, KL={avg_kl:.6f}, '
                  f'β={beta:.3f}, LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    vae.load_state_dict(torch.load('best_enhanced_vae.pth'))
    vae.eval()
    
    return vae, mean_data, std_data, training_history

def vae4dvar():
    print("Generating diverse training data...")
    #training_data = generate_diverse_training_data(V, n_samples=500)  # 减少样本数以便测试
    training_data = generate_data(n_samples=500)
    input_dim = training_data.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # build VAE
    latent_dim = 50  # parameter to adjust
    vae = VAE(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        hidden_dims=[512, 256, 128]
    ).to(device)
    
    # train VAE
    vae, mean_data, std_data, history = train_enhanced_vae(
        vae, training_data, num_epochs=200, batch_size=32  
    )
    vae_optimizer = EnhancedVAE4DVarOptimization(vae, mean_data, std_data, latent_dim)
    z_background = vae_optimizer.encode_function(qb)
    compression_ratio = input_dim / latent_dim

    continue_annotation()

    Z_spaces = [FunctionSpace(mesh, "R", 0) for _ in range(latent_dim)]
    z_funcs = [Function(Z_space, name=f"latent_var_{i}") for i, Z_space in enumerate(Z_spaces)]
    z_background_funcs = [Function(Z_space) for Z_space in Z_spaces]

    for i, (z_func, z_bg_func) in enumerate(zip(z_funcs, z_background_funcs)):
        z_func.dat.data[0] = z_background[i]
        z_bg_func.dat.data[0] = z_background[i]

    def enhanced_vae_decode_function(z_funcs):
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
    #cost function
    alpha_latent = Constant(1e-3)

    J_background_terms = []
    for i in range(latent_dim):
        diff_squared = (z_funcs[i] - z_background_funcs[i])**2
        background_term = alpha_latent * diff_squared * dx
        J_background_terms.append(background_term)
    J_background = J_background_terms[0]
    for term in J_background_terms[1:]:
        J_background = J_background + term
    J_bg_assembled = assemble(J_background)

# traditional: J = J + (assemble((H(solve_rk(q0, return_series=False)) - y_obs)**2 * dx))
# VAE: J = J + (assemble((H(q_decoded) - y_obs)**2 * dx))
    q_decoded = enhanced_vae_decode_function(z_funcs)
    q_final = solve_rk(q_decoded, return_series=False)  #only the last time step, compared with multiple observations
    J_obs_term = assemble((H(q_final) - y_obs)**2 * dx)
    J_total = J_bg_assembled + J_obs_term

    print(f"J_total type: {type(J_total)}")
    print(f"Background cost: {float(J_bg_assembled)}")
    print(f"Observation cost: {float(J_obs_term)}")

    controls = [Control(z_func) for z_func in z_funcs]
    rf_vae = ReducedFunctional(J_total, controls)
    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    z_optimal_funcs = minimize(rf_vae, method="L-BFGS-B", 
                          options={"disp": True, "maxiter": 30}, 
                          derivative_options={'riesz_representation':'l2'})
    #q_vae_opt
    q_vae_optimal = enhanced_vae_decode_function(z_optimal_funcs)
    print("VAE-Var completed successfully")
    return q_vae_optimal

def traditional():
    continue_annotation()
    q0 = qb.copy(deepcopy=True)
    q0.rename("q0")
    
    alpha = Constant(1e-4)
    J = assemble((q0 - qb)**2 * dx)
    J = J + (assemble((H(solve_rk(q0, return_series=False)) - y_obs)**2 * dx))
    J = J + assemble(alpha * (q0 - qb)**2 * dx)
    rf = ReducedFunctional(J, Control(q0))

    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    opt_q0_traditional = minimize(rf, method="L-BFGS-B", 
                                    options={"disp": True, "maxiter": 20}, derivative_options={'riesz_representation':'l2'})
    print(" Traditional 4D-Var optimization completed")
    return opt_q0_traditional

#visualization
def plot_comparison(q_true_init, qb, opt_q0_traditional, q_vae_optimal, observation_points):    
    obs_x = [pt[0] for pt in observation_points]
    obs_y = [pt[1] for pt in observation_points]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced VAE-4D-Var vs Traditional 4D-Var\n(Corrected for Scalar Fields)', 
                 fontsize=16, fontweight='bold')
    
    # q_init
    ax1 = axes[0,0]
    tripcolor(q_true_init, axes=ax1, cmap='RdBu_r', shading='gouraud')
    ax1.set_title(f'q_initial', fontweight='bold')
    ax1.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax1.set_aspect('equal')
    
    # q_b
    ax2 = axes[0,1]
    tripcolor(qb, axes=ax2, cmap='RdBu_r', shading='gouraud')
    ax2.set_title(f'q_b', fontweight='bold')
    ax2.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax2.set_aspect('equal')
    
    # q_traditional
    ax3 = axes[1,0]
    tripcolor(opt_q0_traditional, axes=ax3, cmap='RdBu_r', shading='gouraud')
    ax3.set_title(f'Traditional', fontweight='bold')
    ax3.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax3.set_aspect('equal')
    
    # vae-var
    ax4 = axes[1,1]
    tripcolor(q_vae_optimal, axes=ax4, cmap='RdBu_r', shading='gouraud')
    ax4.set_title(f'VAE-Var', fontweight='bold')
    ax4.scatter(obs_x, obs_y, c='black', s=80, marker='x', linewidth=3)
    ax4.set_aspect('equal')
    
    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig("result graph.png", dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    #setup of the problem
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)
    
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((0.5 - y, x - 0.5))
    u = Function(W).interpolate(velocity)
    bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
    bell = 0.25 * (1 + cos(math.pi * min_value(sqrt(pow(x - bell_x0, 2) + pow(y - bell_y0, 2)) / bell_r0, 1.0)))
    q_true_init = Function(V, name="q_true_init").interpolate(1.0 + bell)
    qb = Function(V, name="qb")
    qb.assign(q_true_init + 0.1 * Function(V).interpolate(sin(2 * math.pi * x) * sin(2 * math.pi * y)))
    
    observation_points = [
        [0.25, 0.5], [0.5, 0.5], [0.75, 0.5], [0.5, 0.25], [0.5, 0.75],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]
    ]
    
    observation_mesh = VertexOnlyMesh(mesh, observation_points)
    vom = FunctionSpace(observation_mesh, "DG", 0)
    
    def H(x):
        return assemble(interpolate(x, vom))
    y_obs = H(q_true_init)
    phi = TestFunction(V)
    dq_trial = TrialFunction(V)

    T = 2*math.pi
    dt = T / 600.0
    dtc = Constant(dt)
    nt = int(T / dt)
    output_freq = 20  
    save_steps = nt // output_freq

    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))
    q_in = Constant(1.0)

    def F1(q):
        return (q * div(phi * u) * dx
          - conditional(dot(u, n) < 0, phi * dot(u, n) * q_in, 0.0) * ds
          - conditional(dot(u, n) > 0, phi * dot(u, n) * q, 0.0) * ds
          - (phi('+') - phi('-')) * (un('+') * q('+') - un('-') * q('-')) * dS)

    a = phi * dq_trial * dx
    def solve_rk(q0_init, return_series=False):
        q = q0_init.copy(deepcopy=True)

        q1, q2 = Function(V), Function(V)
        dq = Function(V)

        L1 = dtc * F1(q)
        L2 = dtc * F1(q1)
        L3 = dtc * F1(q2)

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
        solv1 = LinearVariationalSolver(LinearVariationalProblem(a, L1, dq), solver_parameters=params)
        solv2 = LinearVariationalSolver(LinearVariationalProblem(a, L2, dq), solver_parameters=params)
        solv3 = LinearVariationalSolver(LinearVariationalProblem(a, L3, dq), solver_parameters=params)

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
                qs.append(q.copy(deepcopy=True, annotate=False))

        return qs if return_series else q

    vae_results = vae4dvar()
    tra_results = traditional()
    plot_comparison(q_true_init, qb, tra_results, vae_results, observation_points)
    err_vae = errornorm(q_true_init, vae_results, 'L2')
    err_tra = errornorm(q_true_init, tra_results, 'L2')
    print(f"VAE-Var error: {err_vae:.6f}")
    print(f"Traditional error: {err_tra:.6f}")
    print("successful")
        
   