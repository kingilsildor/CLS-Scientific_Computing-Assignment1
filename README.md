# CLS-Scientific_Computing-Assignment1

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
- [File Descriptions](#file-descriptions)
- [Contributors](#contributors)
- [Git Fame](#git-fame)
- [License](#license)

## Description

Within this repository simulations are created for solving heat diffusion in 2D and the wave equation in 1D.

## Getting Started

### Installation
First clone the repository.
```bash
git clone https://github.com/kingilsildor/CLS-Scientific_Computing-Assignment1
cd repository
```

### Prerequisites

To get the project running, install all the packages from the installer.
For this the following command can be used:
```bash
# Example
pip install -r requirements.txt
```

### Interface
Different modules can be run separately from their file.
But the main inferface for the project is `interface.ipynb` in the root folder.
This file uses all the functions that are important to run the code.

### Style Guide

For controbuting to this project it is important to know the style used in this document.
See the [STYLEGUIDE](STYLEGUIDE.md) file for details.


## File Descriptions

| File/Folder | Description |
|------------|-------------|
| `interface.ipynb` | Interface for all the code |
| `modules/diffusion_equation.py` | File for all the functions related to diffusion |
| `modules/wave_equation.py` | File for all the functions related to the wave equation |
| `data/*` | Store for the data that the functions will write |
| `results/*`| Images and animations of the files |

## Contributors

List all contributors to the project.

- [Tycho Stam](https://github.com/kingilsildor)

## Git Fame

Total commits: 34
Total ctimes: 388
Total files: 28
Total loc: 10466
| Author            |   loc |   coms |   fils |  distribution   |
|:------------------|------:|-------:|-------:|:----------------|
| kingilsildor      | 10323 |     19 |     25 | 98.6/55.9/89.3  |
| Anezka            |   143 |      7 |      3 | 1.4/20.6/10.7   |
| Anezka Potesilova |     0 |      2 |      0 | 0.0/ 5.9/ 0.0   |
| Tycho Stam        |     0 |      6 |      0 | 0.0/17.6/ 0.0   |

Note: Tycho Stam -> kingilsildor

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
