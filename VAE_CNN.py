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

class ConvVAE(nn.Module):
    def __init__(self, input_shape=(1, 80, 80), latent_dim=50, base_channels=32):
        super(ConvVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # 计算编码后的特征图尺寸
        _, h, w = input_shape
        self.encoded_h = h // 16 
        self.encoded_w = w // 16
        self.feature_size = self.encoded_h * self.encoded_w * base_channels * 8
        
        # 编码器
        self.encoder = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层卷积
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层卷积
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层卷积
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 潜在变量映射
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.feature_size)

        self.decoder = nn.Sequential(
            # 第一层反卷积
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层反卷积
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层反卷积
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层
            nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
        print(f"CNN-VAE initialized: {input_shape} -> {latent_dim}D")
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def encode(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        decoded_input = self.decoder_input(z)
        reshaped = decoded_input.view(decoded_input.size(0), self.base_channels*8, 
                                    self.encoded_h, self.encoded_w)
        return self.decoder(reshaped)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, beta=0.5):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def detect_grid_size(function_space):
    test_func = Function(function_space)
    total_size = test_func.dat.data.size
    
    # 寻找最接近的正方形
    side_length = int(np.sqrt(total_size))
    if side_length * side_length == total_size:
        return side_length, side_length
    
    # 如果不是完全平方数，寻找合适的矩形
    for i in range(side_length, 0, -1):
        if total_size % i == 0:
            return i, total_size // i
    
    return side_length, side_length

def generate_training_data(function_space, n_samples=500):
    grid_h, grid_w = detect_grid_size(function_space)
    print(f"Detected grid size: {grid_h} x {grid_w}")
    
    training_data = []
    x, y = SpatialCoordinate(mesh)
    
    for i in range(n_samples):
        noise_field = Function(function_space)
        noise_field.dat.data[:] = np.random.randn(*noise_field.dat.data.shape) * 0.1
        pattern = Function(function_space).interpolate(
            sin(np.random.uniform(1, 4) * math.pi * x) * 
            sin(np.random.uniform(1, 4) * math.pi * y) +
            cos(np.random.uniform(1, 3) * math.pi * x) * 
            cos(np.random.uniform(1, 3) * math.pi * y)
        )
        sample = Function(function_space)
        sample.assign(q_true_init + 0.1 * pattern + 0.05 * noise_field)

        data_1d = sample.dat.data.copy()
        if len(data_1d) == grid_h * grid_w:
            data_2d = data_1d.reshape(1, grid_h, grid_w)
        else:
            target_size = grid_h * grid_w
            if len(data_1d) > target_size:
                data_1d = data_1d[:target_size]
            else:
                padded = np.zeros(target_size)
                padded[:len(data_1d)] = data_1d
                data_1d = padded
            data_2d = data_1d.reshape(1, grid_h, grid_w)
        
        training_data.append(data_2d)
    
    return np.array(training_data), grid_h, grid_w

def train_vae(vae, training_data, epochs=200, batch_size=16, lr=1e-4):
    device = next(vae.parameters()).device

    mean_data = np.mean(training_data, axis=0)
    std_data = np.std(training_data, axis=0) + 1e-8
    normalized_data = (training_data - mean_data) / std_data

    dataset = TensorDataset(torch.FloatTensor(normalized_data), 
                           torch.FloatTensor(normalized_data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"Training VAE for {epochs} epochs...")
    vae.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar, _ = vae(batch_data)
            
            # β退火
            beta = min(1.0, (epoch + 1) / 50)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, logvar, beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), 'best_cnn_vae.pth')
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss = {avg_loss:.6f}, β = {beta:.3f}')
    vae.load_state_dict(torch.load('best_cnn_vae.pth'))
    vae.eval()
    
    return vae, mean_data, std_data

class CNNVAEAdapter:
    def __init__(self, vae, mean_data, std_data, grid_h, grid_w):
        self.vae = vae
        self.mean_data = mean_data
        self.std_data = std_data
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.device = next(vae.parameters()).device
    
    def function_to_tensor(self, func):
        data_1d = func.dat.data_ro.copy()
        target_size = self.grid_h * self.grid_w
        if len(data_1d) != target_size:
            if len(data_1d) > target_size:
                data_1d = data_1d[:target_size]
            else:
                padded = np.zeros(target_size)
                padded[:len(data_1d)] = data_1d
                data_1d = padded
        
        data_2d = data_1d.reshape(1, self.grid_h, self.grid_w)
        normalized = (data_2d - self.mean_data) / self.std_data
        
        return torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
    
    def tensor_to_function(self, tensor, func_space):
        data_2d = tensor.squeeze().cpu().detach().numpy()
        denormalized = data_2d * self.std_data + self.mean_data
        data_1d = denormalized.flatten()
        
        func = Function(func_space)
        func_size = func.dat.data.size
        
        if len(data_1d) >= func_size:
            func.dat.data[:] = data_1d[:func_size]
        else:
            padded = np.zeros(func_size)
            padded[:len(data_1d)] = data_1d
            func.dat.data[:] = padded
        
        return func
    
    def encode_function(self, func):
        tensor = self.function_to_tensor(func)
        with torch.no_grad():
            mu, _ = self.vae.encode(tensor)
            return mu.squeeze().cpu().numpy()
    
    def decode_latent(self, z_array, func_space):
        z_tensor = torch.FloatTensor(z_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.vae.decode(z_tensor)
            return self.tensor_to_function(decoded, func_space)

def cnn_vae_4dvar():
    training_data, grid_h, grid_w = generate_training_data(V, n_samples=500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = ConvVAE(input_shape=(1, grid_h, grid_w), latent_dim=50).to(device)
    vae, mean_data, std_data = train_vae(vae, training_data)

    adapter = CNNVAEAdapter(vae, mean_data, std_data, grid_h, grid_w)
    z_background = adapter.encode_function(qb)
    print(f"Compression: {grid_h*grid_w}D -> {len(z_background)}D")
    
    continue_annotation()

    latent_dim = len(z_background)
    Z_spaces = [FunctionSpace(mesh, "R", 0) for _ in range(latent_dim)]
    z_funcs = [Function(space) for space in Z_spaces]
    z_bg_funcs = [Function(space) for space in Z_spaces]

    for i, (z_func, z_bg_func) in enumerate(zip(z_funcs, z_bg_funcs)):
        z_func.dat.data[0] = z_background[i]
        z_bg_func.dat.data[0] = z_background[i]
    
    def decode_function(z_funcs_list):

        z_array = np.array([zf.dat.data[0] for zf in z_funcs_list])
        pause_annotation()
        decoded_func = adapter.decode_latent(z_array, V)
        continue_annotation()
        return decoded_func
    alpha = Constant(1e-3)

    J_bg = sum(alpha * (zf - zbf)**2 * dx for zf, zbf in zip(z_funcs, z_bg_funcs))
    J_bg_assembled = assemble(J_bg)

    q_decoded = decode_function(z_funcs)
    q_final = solve_rk(q_decoded, return_series=False)
    J_obs = assemble((H(q_final) - y_obs)**2 * dx)

    J_total = J_bg_assembled + J_obs
    
    print(f"Background cost: {float(J_bg_assembled):.6f}")
    print(f"Observation cost: {float(J_obs):.6f}")

    controls = [Control(zf) for zf in z_funcs]
    rf = ReducedFunctional(J_total, controls)
    
    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    z_optimal = minimize(rf, method="L-BFGS-B", 
                        options={"disp": True, "maxiter": 30},derivative_options={'riesz_representation':'l2'})

    q_optimal = decode_function(z_optimal)
    print("CNN-VAE 4D-Var completed!")
    
    return q_optimal

def traditional_4dvar():
    continue_annotation()
    q0 = qb.copy(deepcopy=True)
    
    alpha = Constant(1e-4)
    J = assemble((q0 - qb)**2 * dx)
    J += assemble((H(solve_rk(q0, return_series=False)) - y_obs)**2 * dx)
    J += assemble(alpha * (q0 - qb)**2 * dx)
    
    rf = ReducedFunctional(J, Control(q0))
    
    pause_annotation()
    get_working_tape().progress_bar = ProgressBar
    q_optimal = minimize(rf, method="L-BFGS-B", 
                        options={"disp": True, "maxiter": 20}, derivative_options={'riesz_representation':'l2'})
    
    print("Traditional 4D-Var completed!")
    return q_optimal

def plot_results(q_true, q_background, q_traditional, q_cnn_vae, obs_points):
    obs_x = [pt[0] for pt in obs_points]
    obs_y = [pt[1] for pt in obs_points]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CNN-VAE 4D-Var vs Traditional 4D-Var', fontsize=16, fontweight='bold')

    tripcolor(q_true, axes=axes[0,0], cmap='RdBu_r')
    axes[0,0].set_title('True Initial Field')
    axes[0,0].scatter(obs_x, obs_y, c='black', s=80, marker='x')

    tripcolor(q_background, axes=axes[0,1], cmap='RdBu_r')
    axes[0,1].set_title('Background Field')
    axes[0,1].scatter(obs_x, obs_y, c='black', s=80, marker='x')

    tripcolor(q_traditional, axes=axes[1,0], cmap='RdBu_r')
    axes[1,0].set_title('Traditional 4D-Var')
    axes[1,0].scatter(obs_x, obs_y, c='black', s=80, marker='x')

    tripcolor(q_cnn_vae, axes=axes[1,1], cmap='RdBu_r')
    axes[1,1].set_title('CNN-VAE 4D-Var')
    axes[1,1].scatter(obs_x, obs_y, c='black', s=80, marker='x')
    
    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('cnn_vae_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)
    
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((0.5 - y, x - 0.5))
    u = Function(W).interpolate(velocity)
    bell_r0 = 0.15
    bell_x0 = 0.25
    bell_y0 = 0.5
    bell = 0.25 * (1 + cos(math.pi * min_value(sqrt(pow(x - bell_x0, 2) + pow(y - bell_y0, 2)) / bell_r0, 1.0)))
    q_true_init = Function(V, name="q_true").interpolate(1.0 + bell)

    qb = Function(V, name="qb")
    qb.assign(q_true_init + 0.1 * Function(V).interpolate(sin(2 * math.pi * x) * sin(2 * math.pi * y)))
    
    # 观测点
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
    T = 2 * math.pi
    dt = T / 600.0
    dtc = Constant(dt)
    
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
        
        solv1 = LinearVariationalSolver(LinearVariationalProblem(a, L1, dq, constant_jacobian=True), solver_parameters=params)
        solv2 = LinearVariationalSolver(LinearVariationalProblem(a, L2, dq, constant_jacobian=True), solver_parameters=params)
        solv3 = LinearVariationalSolver(LinearVariationalProblem(a, L3, dq, constant_jacobian=True), solver_parameters=params)
        
        qs = []
        t, step = 0.0, 0
        output_freq = 20
        
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
    q_cnn_vae = cnn_vae_4dvar()
    q_traditional = traditional_4dvar()

    err_cnn_vae = errornorm(q_true_init, q_cnn_vae, 'L2')
    err_traditional = errornorm(q_true_init, q_traditional, 'L2')
    print(f"CNN-VAE 4D-Var L2 error:     {err_cnn_vae:.6f}")
    print(f"Traditional 4D-Var L2 error: {err_traditional:.6f}")
    plot_results(q_true_init, qb, q_traditional, q_cnn_vae, observation_points)
