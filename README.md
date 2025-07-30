# Frameon

[![PyPI Versions](https://img.shields.io/pypi/v/frameon?logo=PyPI)](https://pypi.org/project/frameon)
[![Python Versions](https://img.shields.io/pypi/pyversions/frameon.svg)](https://pypi.org/project/frameon/)
[![Documentation Status](https://readthedocs.org/projects/frameon/badge/?version=latest)](https://frameon.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/PavelGrigoryevDS/frameon/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/PavelGrigoryevDS/frameon/actions)
[![codecov](https://codecov.io/gh/PavelGrigoryevDS/frameon/branch/main/graph/badge.svg)](https://codecov.io/gh/PavelGrigoryevDS/frameon)
[![Dependabot](https://img.shields.io/badge/dependabot-active-brightgreen.svg)](https://github.com/PavelGrigoryevDS/frameon/network/updates)
[![License](https://img.shields.io/pypi/l/frameon.svg)](https://opensource.org/licenses/MIT)

Frameon extends pandas DataFrame with analysis methods while keeping all original functionality intact.

⭐ **If you find Frameon useful, please star the GitHub repo!** It helps others discover the project and motivates further development.

---

## ✨ Key Features

- **Seamless integration**: Works with existing pandas DataFrames and Series
- **Non-intrusive**: All pandas methods remain unchanged and fully available
- **Modular access**: Additional functionality organized in clear namespaces
- **Dual-level access**: Methods available for both entire DataFrames and individual columns

---

## 📦 Installation

❗ **Recommended:** Use a virtual environment to prevent potential conflicts with existing package versions in your system. 

### Basic Installation

#### Using pip
```bash
pip install frameon
```

#### Using poetry
```bash
poetry add frameon
```

### Installation with Virtual Environment

#### Python's built-in venv
```bash
# Create virtual environment
python -m venv frameon_env

# Activate it
source frameon_env/bin/activate 

# Install frameon
pip install frameon
```

#### Using poetry (manages virtual env automatically)
```bash
# Navigate to your project directory
poetry init  # if starting new project
poetry add frameon
```

---

## 🚀 Quick Start

```python
import pandas as pd
from frameon import FrameOn as fo

# Create or load your DataFrame
df = pd.read_csv('your_data.csv')

# Add Frameon functionality
df = fo(df)

# Explore your data
df.explore.info()           # For entire DataFrame
df['price'].explore.info()  # For individual column
```

---

## 📚 Documentation

For complete documentation including API reference and more examples, visit:  
[Frameon Documentation](https://frameon.readthedocs.io/en/latest/)

---

## 🧪 Examples

### Data Exploration

Quickly explore your data:

```python
from frameon import load_dataset, FrameOn as fo

titanic = fo(load_dataset('titanic'))
titanic['age'].explore.info()
```
<img src="https://raw.githubusercontent.com/PavelGrigoryevDS/frameon/main/images/info.png" width="600">

---

### Cohort Analysis

Quickly visualize user retention with a cohort heatmap:

```python
from frameon import load_dataset, FrameOn as fo

superstore = fo(load_dataset('superstore'))
fig = superstore.analysis.cohort(
    user_id_col='Customer ID', 
    date_col='Order Date', 
    revenue_col='Sales',
    granularity='quarter',
    include_period0=False,
)
fig.show()
```

<img src="https://raw.githubusercontent.com/PavelGrigoryevDS/frameon/main/images/cohort.png" width="800">

---

### Statistical Tests

Compare groups using bootstrap:

```python
from frameon import load_dataset, FrameOn as fo

titanic = fo(load_dataset('titanic'))
titanic.stats.bootstrap(
    dv='age',
    between='alive',
    reference_group='no',
    statistic='mean_diff',
    plot=True
)
```

<img src="https://raw.githubusercontent.com/PavelGrigoryevDS/frameon/main/images/bootstrap.png" width="700">

---

## 🔧 API Overview

Frameon provides methods through these namespaces:

| Namespace   | Description                          | DataFrame | Series |
|-------------|--------------------------------------|-----------|--------|
| `.explore`  | Data exploration and quality checks  | ✓         | ✓      |
| `.preproc`  | Data preprocessing and cleaning      | ✓         | ✓      |
| `.analysis` | Advanced analytical methods          | ✓         | ✗      |
| `.viz`      | Visualization methods                | ✓         | ✗      |
| `.stats`    | Statistical tests and analysis       | ✓         | ✗      |

---

## ⚙️ Built With

Frameon utilizes these foundational libraries:

- [pandas](https://pandas.pydata.org/) - Core data structures
- [numpy](https://numpy.org/) - Numerical computing
- [plotly](https://plotly.com/python/) - Interactive visualization
- [scipy](https://www.scipy.org/) - Scientific computing
- [statsmodels](https://www.statsmodels.org/) - Statistical modeling
- [pingouin](https://pingouin-stats.org/) - Statistics
- [scikit-learn](https://scikit-learn.org/) - Machine learning

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. 🐛 Report bugs via [GitHub Issues](https://github.com/PavelGrigoryevDS/frameon/issues)
2. 📥 Submit PRs for new features
3. 📖 Improve documentation

See our [Contributing Guidelines](https://github.com/PavelGrigoryevDS/frameon/blob/main/CONTRIBUTING.md) for details.

---

## 📜 License

Frameon is licensed under the [MIT License](https://github.com/PavelGrigoryevDS/frameon/blob/main/LICENSE).
