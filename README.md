# NOIR: Neural Operator Independence Regularization

Official implementation of "Neural Operator Independence Regularization" 

## 🗂️ Repository Structure

```
NOIR/
├── src/                          # Core implementation
│   ├── fcn.py                   # Fully Connected Network
│   ├── deeponet.py              # DeepONet architecture
│   ├── noir.py                  # NOIR variants (Orthogonal DeepONet & FNO)
│   ├── fno.py                   # Fourier Neural Operator
│   ├── wno.py                   # Wavelet Neural Operator
│   ├── nomad.py                 # NOMAD architecture
│   ├── losses.py                # Loss functions including ICA loss
│   ├── utils.py                 # Utility functions
│   └── noise_generators.py     # Structured noise generation
│
├── experiments/
│   ├── naca/                    # NACA airfoil experiments
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── utils_naca.py
│   │   └── visualize.py
│   │
│   ├── darcy/                   # Darcy flow experiments
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── utils_darcy.py
│   │   └── visualize.py
│   │
│   └── burgers/                 # Burgers equation experiments
│       ├── config.yaml
│       ├── train.py
│       ├── evaluate.py
│       └── visualize.py
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── quick_demo.ipynb
│
├── results/                      # Output directory (auto-created)
│   ├── models/                  # Trained models
│   ├── logs/                    # Training logs
│   └── figures/                 # Generated figures
│
├── requirements.txt
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- PyTorch 1.9+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NOIR.git
cd NOIR
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Datasets

### Download Links
- **NACA Airfoil**: [Download](https://drive.google.com/drive/folders/)
- **Darcy Flow**: [Download](https://drive.google.com/drive/folders/) 
- **Burgers Equation**: [Download](https://drive.google.com/drive/folders/)

### Dataset Preparation
1. Download the datasets from the links above
2. Place them in the appropriate directories as specified in config files
3. Update the `data_path` in each experiment's `config.yaml`

## 🚀 Running Experiments

### Training Models

For each experiment (naca/darcy/burgers):

```bash
cd experiments/[experiment_name]
python train.py --config config.yaml
```

This will train all model variants with multiple seeds as specified in the configuration.

### Evaluating Noise Robustness

After training, evaluate models under various noise conditions:

```bash
python evaluate.py --config config.yaml
```

### Generating Figures

Create publication-ready figures and tables:

```bash
python visualize.py --config config.yaml
```

### Example: Complete Pipeline for NACA

```bash
cd experiments/naca

# Train all models (DeepONet, NOMAD, FNO, WNO, and NOIR variants)
python train.py

# Evaluate noise robustness
python evaluate.py

# Generate figures
python visualize.py
```

## 📝 Configuration

Each experiment uses a YAML configuration file with the following structure:

```yaml
data:
  data_path: /path/to/data
  ntrain: 1000
  ntest: 200
  batch_size: 32

models:
  types_don: ['default', 'nomad', 'ortho_ica']
  types_fno: ['fourier', 'fourier_ortho']

hyperparameters:
  width_don: 128
  width_fno: 64
  modes: 12
  ica_dim: 256

training:
  seeds: [0, 1, 42]
  epochs: 500
  learning_rate: 1e-3
  gamma: 0.001  # orthogonality weight

noise_testing:
  types: ['partial_random', 'harmonic_spatial']
  levels: [0.0, 0.1, 0.5, 1.0]
```

## 📊 Notebooks

Interactive Jupyter notebooks are provided for:
- Data exploration and visualization
- Model architecture inspection
- Noise generation demonstration
- Result analysis and plotting


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the authors of DeepONet, FNO, and NOMAD for their foundational work
- Datasets provided by [respective sources]