# Neural Networks Meet Physics: A Deep Dive into PINNs

## What's This All About?

Ever wondered what happens when you teach neural networks to respect the laws of physics? Well, that's exactly what this thesis explores! Instead of just throwing data at a neural network and hoping for the best, Physics-Informed Neural Networks (PINNs) actually embed physical laws directly into the learning process.

Think of it this way: imagine you're trying to predict how heat spreads through a metal rod. Traditional neural networks would need tons of temperature measurements. But PINNs? They just need to know the heat equation, and they can figure out the rest. Pretty neat, right?

---

## The Journey Through Mathematics

This project takes you on a wild ride through the mathematical landscape:

**Starting Simple with ODEs**
- Basic exponential decay (like radioactive materials cooling down)
- Population growth models (how rabbits multiply... or don't)
- Van der Pol oscillators (think heartbeats and electronic circuits)

**Diving into PDEs** 
- Heat diffusion (ever wonder why your coffee gets cold?)
- Wave propagation (sound, vibrations, guitar strings!)
- Burgers' equation (fluid dynamics meets real-world chaos)

**Getting Fancy with Advanced Experiments**
- Battle of the optimizers: Adam vs L-BFGS (spoiler: it depends!)
- Network architecture deep-dive (bigger isn't always better)
- Noise tolerance testing (because real data is messy)
- Detective work: finding hidden parameters

## How Everything Fits Together

```
PINN/
├── code/                     # Where the magic happens
│   ├── pinn_base.py         # The foundation (like a blueprint)
│   ├── ode_solver.py        # Simple differential equations
│   ├── pde_solver.py        # The heavy-duty physics stuff
│   ├── experiments.py       # Advanced wizardry and comparisons
│   └── thesis_runner.py     # The conductor of this orchestra
├── tests/                   # Making sure nothing breaks
│   └── test_pinn_thesis.py  # 21 tests that better all pass!
├── figures/                 # Pretty pictures and graphs
├── data/                    # Currently empty (PINNs are data-light!)
├── docs/                    # Extra reading material
└── README.md               # You are here
```

## Getting Your Hands Dirty

### What You'll Need

- Python 3.8+ (because we're not living in the stone age)
- A CUDA GPU helps, but your CPU can handle the smaller experiments

### The Essential Toolkit

Fire up your terminal and grab these packages:

```bash
pip install torch numpy matplotlib seaborn
```

Want to play around and run tests? Add these too:

```bash
pip install pytest jupyter notebook
```

### Ready, Set, Code!

1. Grab this repository (download or clone, your choice)
2. Navigate to the project folder
3. Install the packages above
4. You're officially ready to bend physics to your will!

## Let's Fire This Thing Up!

### The Full Experience (Buckle Up!)

Want to see everything this project can do? Run the complete show:

```bash
cd code
python thesis_runner.py --all
```

First time? Start with the quick tour (trust me, it's still impressive):

```bash
python thesis_runner.py --all --quick
```

### Pick Your Adventure

Not feeling the full commitment? Choose your own path:

```bash
# Just the ODEs (start here if you're new to this)
python thesis_runner.py --ode

# PDEs only (where things get spicy)
python thesis_runner.py --pde

# The advanced stuff (for when you're feeling brave)
python thesis_runner.py --experiments

# Make sure nothing's broken
python thesis_runner.py --tests
```

## How to Use the Code

The `/code` directory contains all the implementation files. Here's how to use each component:

### Core Framework Files

#### `pinn_base.py` - The Foundation
This contains the base classes that everything else builds on:

```python
from pinn_base import BasePINN, PINNNet

# Create a neural network for PINNs
network = PINNNet(input_dim=2, hidden_dim=64, hidden_layers=4, activation='tanh')

# All PINN solvers inherit from BasePINN
# You can extend it to create your own physics problems
```

#### `ode_solver.py` - Ordinary Differential Equations
Contains solvers for time-dependent problems:

```python
from ode_solver import LinearODEPINN, LogisticODEPINN, VanDerPolPINN, run_ode_experiment

# Method 1: Use the pre-built experiment runner
solver, history = run_ode_experiment('linear', t_max=3.0, u0=1.0, lambda_coeff=1.0)

# Method 2: Create and train your own solver
linear_solver = LinearODEPINN(t_max=5.0, u0=2.0, lambda_coeff=0.5)
history = linear_solver.train(n_iterations=5000, learning_rate=1e-3)

# Get predictions
import numpy as np
t_test = np.linspace(0, 5, 100).reshape(-1, 1)
solution = linear_solver.predict(t_test)

# Compare with exact solution
exact = linear_solver.exact_solution(t_test)
error = linear_solver.compute_relative_l2_error(t_test, exact)
print(f"Relative L2 error: {error:.6f}")
```

#### `pde_solver.py` - Partial Differential Equations
Handles space-time problems:

```python
from pde_solver import HeatEquationPINN, WaveEquationPINN, BurgersEquationPINN, run_pde_experiment

# Method 1: Quick experiment
solver, history = run_pde_experiment('heat', x_max=1.0, t_max=0.5, alpha=0.1)

# Method 2: Manual setup for custom control
heat_solver = HeatEquationPINN(x_max=2.0, t_max=1.0, alpha=0.05, initial_condition='gaussian')
history = heat_solver.train(
    n_iterations=10000,
    learning_rate=1e-3,
    n_collocation=2000,
    lambda_ic=10.0,
    lambda_bc=10.0,
    optimizer_type='adam'
)

# Generate solution on a grid
x = np.linspace(0, 2, 50)
t = np.linspace(0, 1, 50)
X, T = np.meshgrid(x, t)
xt_grid = np.column_stack([X.flatten(), T.flatten()])
solution = heat_solver.predict(xt_grid).reshape(50, 50)
```

#### `experiments.py` - Advanced Studies
For comparative studies and parameter sweeps:

```python
from experiments import OptimizerComparison, ArchitectureStudy, NoiseRobustnessStudy

# Compare optimizers
optimizer_study = OptimizerComparison(problem_type='linear_ode')
results = optimizer_study.run_comparison(['adam', 'lbfgs'], n_iterations=5000)
optimizer_study.plot_comparison('optimizer_comparison.png')

# Study network architecture effects
arch_study = ArchitectureStudy('heat_pde')
width_results = arch_study.width_study([16, 32, 64, 128])
depth_results = arch_study.depth_study([2, 3, 4, 5, 6])
arch_study.plot_results('architecture_study.png')

# Test noise robustness
noise_study = NoiseRobustnessStudy('logistic_ode')
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
results = noise_study.run_noise_study(noise_levels)
noise_study.plot_results('noise_robustness.png')
```

#### `thesis_runner.py` - The Main Controller
Orchestrates all experiments:

```python
# Run from command line:
# python thesis_runner.py --all --quick
# python thesis_runner.py --ode
# python thesis_runner.py --pde --save-dir custom_results/

# Or use programmatically:
from thesis_runner import run_all_experiments, run_ode_experiments, run_pde_experiments

# Run specific experiment sets
ode_results = run_ode_experiments(quick=True, save_dir='results/')
pde_results = run_pde_experiments(quick=False, save_dir='results/')
```

### Creating Your Own PINN Solver

To implement a new physics problem:

```python
from pinn_base import BasePINN, PINNNet
import torch
import numpy as np

class MyCustomPINN(BasePINN):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
        
        # Define your network architecture
        self.model = PINNNet(
            input_dim=1,      # Change based on your problem
            hidden_dim=32,
            hidden_layers=3,
            output_dim=1
        ).to(self.device)
    
    def compute_pde_residual(self, x):
        """Define your differential equation here"""
        u = self.model(x)
        
        # Compute derivatives using automatic differentiation
        du_dx = torch.autograd.grad(
            u, x, torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # Your PDE: u' = -custom_param * u
        residual = du_dx + self.custom_param * u
        return residual
    
    def compute_ic_loss(self):
        """Define initial conditions"""
        x0 = torch.zeros(1, 1, device=self.device)
        u0_pred = self.model(x0)
        # u(0) = 1
        ic_loss = torch.mean((u0_pred - 1.0) ** 2)
        return ic_loss
    
    def generate_collocation_points(self, n_points):
        """Generate points where we check the physics"""
        x = np.random.uniform(0, 5, (n_points, 1))
        return self.to_tensor(x, requires_grad=True)

# Use your custom solver
my_solver = MyCustomPINN(custom_param=2.0)
history = my_solver.train(n_iterations=3000)
```

### Customizing Training Parameters

All solvers support extensive customization:

```python
# Training parameters you can adjust
training_config = {
    'n_iterations': 10000,        # How long to train
    'learning_rate': 1e-3,        # Step size for optimization
    'n_collocation': 1000,        # How many physics points to check
    'lambda_ic': 10.0,            # Weight for initial condition loss
    'lambda_bc': 1.0,             # Weight for boundary condition loss
    'optimizer_type': 'adam',     # 'adam' or 'lbfgs'
    'print_frequency': 500        # How often to print progress
}

history = solver.train(**training_config)
```

### Working with Results

All solvers return training history and provide utilities:

```python
# Training history contains loss components
print(f"Final total loss: {history['total_loss'][-1]}")
print(f"Final PDE loss: {history['pde_loss'][-1]}")

# Generate predictions
test_points = np.linspace(0, 1, 100).reshape(-1, 1)
predictions = solver.predict(test_points)

# Compute errors (if exact solution available)
if hasattr(solver, 'exact_solution'):
    exact = solver.exact_solution(test_points)
    error = solver.compute_relative_l2_error(test_points, exact)
    print(f"Relative L2 error: {error:.6f}")

# Plot training curves
from pinn_base import save_training_plots
save_training_plots(history, "my_training.png")
```

## The Experimental Playground

### Chapter 1: Ordinary Differential Equations (The Warm-Up)

#### The Classic Linear Decay
- **What it does**: `u'(t) = -λu(t)`, starting at `u(0) = u₀`
- **Real-world meaning**: Coffee cooling, radioactive decay, your enthusiasm for homework
- **The beauty**: We know the exact answer: `u(t) = u₀e^(-λt)`
- **Why it matters**: Perfect for testing if PINNs actually work

#### Population Dynamics (The S-Curve Story)
- **The math**: `u'(t) = ru(1-u)`, beginning at `u(0) = u₀`
- **In plain English**: How populations grow when resources are limited
- **Reality check**: Starts slow, explodes, then levels off (like viral TikTok videos)
- **The exact formula**: `u(t) = 1/(1 + ((1-u₀)/u₀)e^(-rt))` (don't memorize this)

#### Van der Pol Oscillator (Where Things Get Weird)
- **The beast**: `u'' - μ(1-u²)u' + u = 0`
- **Think of it as**: A self-regulating oscillation (like your heart or an old radio)
- **Cool factor**: Creates those mesmerizing limit cycle patterns
- **Difficulty level**: Medium spicy

### Chapter 2: Partial Differential Equations (The Real Deal)

#### Heat Diffusion (Your Morning Coffee)
- **The physics**: `u_t = αu_xx` 
- **Domain**: From `x = 0` to `x = L`, over time `t`
- **Boundary rules**: Temperature fixed at both ends
- **What you'll see**: Heat spreading and evening out over time
- **Fun fact**: This is why your coffee gets cold (sorry)

#### Wave Propagation (Good Vibrations)
- **The equation**: `u_tt = c²u_xx`
- **What it models**: Sound waves, guitar strings, seismic activity
- **The twist**: Two time derivatives make this trickier than heat
- **Coolness factor**: You can literally see waves bouncing around

#### Burgers' Equation (Fluid Chaos)
- **The formula**: `u_t + uu_x = νu_xx`
- **Why it's special**: Combines convection AND diffusion
- **Reality**: Simplified fluid dynamics that still captures shock formation
- **Warning**: This one fights back!

### Chapter 3: Advanced Experiments (The Graduate Level Stuff)

#### The Great Optimizer Showdown
- **Contenders**: Adam (the reliable workhorse) vs L-BFGS (the sophisticated speedster)
- **What we measure**: Speed, accuracy, and who gives up first
- **Spoiler alert**: Each has their strengths depending on the problem

#### Network Architecture Investigation
- **Width experiment**: How many neurons per layer? (16 to 256)
- **Depth study**: How many layers deep should we go? (2 to 8)
- **Activation face-off**: tanh vs ReLU vs sigmoid
- **Key insight**: More isn't always better (shocking, I know)

#### Noise Tolerance Testing
- **The challenge**: What happens when initial conditions are messy?
- **Real-world relevance**: Because perfect data doesn't exist
- **The question**: How much noise can PINNs handle before they throw in the towel?

## What to Expect (Performance-wise)

### The Report Card

| Problem | Error Range | Time to Train | Behavior |
|---------|-------------|---------------|----------|
| Linear ODE | 10⁻⁴ to 10⁻⁶ | 10-30 seconds | A+ student, always converges |
| Logistic ODE | 10⁻³ to 10⁻⁵ | 15-45 seconds | Solid B+, occasional hiccup |
| Heat PDE | 10⁻² to 10⁻⁴ | 1-3 minutes | Steady worker, reliable |
| Wave PDE | 10⁻² to 10⁻³ | 1.5-5 minutes | Moody but gets there |
| Burgers PDE | 10⁻² to 10⁻¹ | 3-10 minutes | The challenging one |

### What You'll Get

Every experiment spits out:
- **Gorgeous plots**: PINN solutions vs the "right" answers (when we know them)
- **Training diaries**: How the loss function behaved during learning
- **Error breakdowns**: Just how wrong (or right) we were
- **Head-to-head comparisons**: Different approaches duking it out

All the visual goodies land in the `figures/` folder with names that actually make sense.

## Testing (Because Nobody Likes Broken Code)

### Making Sure Everything Works

```bash
# Run the full test battery
python thesis_runner.py --tests

# Or go direct if you're feeling rebellious
cd tests
python test_pinn_thesis.py
```

### What Gets Tested

Our 21 tests make sure that:
- The PINN foundation doesn't crumble
- All ODE and PDE solvers actually solve things
- Training doesn't go off the rails
- Error calculations aren't lying to us
- Experiments run without exploding
- Everything works end-to-end

### The Testing Philosophy

These tests are built to:
- Finish before you lose patience (< 5 minutes)
- Work whether you have a fancy GPU or just a CPU
- Catch mathematical mistakes before they embarrass you
- Make sure results are repeatable (because science)

## Under the Hood (For the Curious)

### How PINNs Actually Work

1. **Neural Network Magic**: Use multi-layer perceptrons to guess the solution `u(x,t)`
2. **Automatic Differentiation**: Let PyTorch compute derivatives (no manual calculus!)
3. **Physics-Informed Loss**: Mix PDE residuals with boundary/initial conditions
4. **Smart Optimization**: Train using gradient-based methods (Adam or L-BFGS)

### The Loss Function Recipe

```
Total Loss = λ₁ × PDE_Loss + λ₂ × IC_Loss + λ₃ × BC_Loss
```

Breaking it down:
- **PDE_Loss**: How badly we're violating the differential equation
- **IC_Loss**: How far off we are from initial conditions
- **BC_Loss**: How much we're ignoring boundary conditions

### What Makes This Implementation Special

- **Plugin Architecture**: Add new problems without breaking existing code
- **GPU-Ready**: Automatically uses your graphics card if available
- **Reproducible**: Set seeds once, get same results forever
- **Auto-Visualization**: Pretty plots generated automatically
- **Error Tracking**: Built-in accuracy measurement against known solutions

### Network Design Choices

The default neural network setup:
- **Input Layer**: 1D for ODEs, 2D for PDEs (makes sense, right?)
- **Hidden Layers**: 3-4 layers with 32-64 neurons (sweet spot for most problems)
- **Activation**: Hyperbolic tangent (smooth and well-behaved)
- **Output**: Single value (the solution at each point)
- **Weight Initialization**: Xavier uniform (fancy way to start with good guesses)

## Why This Matters (The Big Picture)

This codebase isn't just homework—it's a complete learning journey:

1. **Training Wheels for PINNs**: Start here if you're new to physics-informed learning
2. **Research Springboard**: Solid foundation for your own wild ideas
3. **Benchmark Central**: Standard problems to test new methods against
4. **Code That Actually Works**: No prototype spaghetti code here

### If You're a Student

- Everything's explained (even the obvious stuff)
- Starts simple, gets progressively more interesting
- Handles errors gracefully (because we've all been there)
- Tests ensure you're not building on quicksand

### If You're a Researcher

- Clean, extensible design (add your own problems easily)
- Drop in new experiments without breaking everything
- Hyperparameter studies built-in
- Consistent evaluation metrics across all problems

## Want to Contribute? Here's How

### Adding Your Own Physics Problem

1. Inherit from the `BasePINN` class (it's your friend)
2. Implement the required methods:
   - `compute_pde_residual()` - Define your physics
   - `compute_ic_loss()` - Set initial conditions
   - `generate_collocation_points()` - Where to sample
3. Know the exact solution? Add it for validation
4. Write tests (future you will thank you)

### Extending the Experiments

1. Add new classes to `experiments.py`
2. Follow the existing patterns (consistency is king)
3. Handle errors gracefully and visualize results
4. Update tests so everything stays working

## License & Legal Stuff

This is educational code from a bachelor thesis. Use it, modify it, learn from it. Just don't blame me if your research takes an unexpected turn!

## Standing on the Shoulders of Giants

The foundational papers that made this all possible:

1. **Raissi et al. (2019)** - The original PINN paper that started it all
   *Journal of Computational Physics*, 378, 686-707

2. **Karniadakis et al. (2021)** - The big picture view of physics-informed ML
   *Nature Reviews Physics*, 3(6), 422-440

3. **Cuomo et al. (2022)** - Where we are and where we're going
   *Journal of Scientific Computing*, 92(3), 1-62

---

**Created by**: A curious bachelor student who got way too excited about neural networks  
**Year**: 2025  
**Status**: Actually works! (Tested on real computers)

Got questions? Found a bug? Want to chat about PINNs? Don't hesitate to reach out!
# PINN_DE_playground
