# MTPHandler

[![Documentation](https://img.shields.io/badge/Documentation-Online-blue.svg)](https://fairchemistry.github.io/MTPHandler/)
[![Tests](https://github.com/FAIRChemistry/MTPHandler/actions/workflows/tests.yml/badge.svg)](https://github.com/FAIRChemistry/MTPHandler/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/mtphandler.svg)](https://badge.fury.io/py/mtphandler)


## ℹ️ Overview

`mtphandler` is a tool for managing and processing data from microtiter plates. It allows direct reading of output files from various photometers, enabling low-friction data handling. The tool facilitates a workflow for importing raw data, assigning molecules with their respective concentrations and units to wells.

Wells for creating standard curves can be automatically detected and fitted to calibration models, which are then used to calculate the concentration of unknown samples. Finally, the plate data can be transformed into time-course concentration data in the EnzymeML format, enabling downstream analysis.


``` mermaid
graph LR
  AP[🧪 Plate Reader] --> A[📄 Output File];
  style AP fill:transparent,stroke:#000,stroke-width:2px;
  A -->|read| B{mtphandler}
  style B stroke-width:4px
  subgraph in Jupyter Notebook
    subgraph with mtphandler
        B --> B1[Enrich Data with Metadata]
        B1 --> B2[Blank Data]
        B2 --> B3[Create and Apply Calibration Models]
        B3 --> B

        style B1 stroke-dasharray: 5, 5
        style B2 stroke-dasharray: 5, 5
        style B3 stroke-dasharray: 5, 5
    end
  B -->|convert| G[📄 EnzymeML time-course Data]
  G <-.-> H[📊 Data Science and Insights]

  style H stroke-dasharray: 5, 5,fill:transparent
  end
  G -->|export| I[📄 EnzymeML File]
```

## ⭐ Key Features

- **🚀 Parser Functions**  
   Features a custom parser for various plate readers, enabling low-friction data processing.

- **🌟 Enrich measured data with metadata**  
    Assigns molecules with their respective concentration and unit to wells, capturing the experimental context of each well.

- **⚙️ Adaptive Data Processing**  
   Automatically adapts and blanks measurement data based on initial conditions set for each well. Automatically classifies wells without protein as calibration data and those with protein as reaction data.

- **🌐 FAIR Data**  
   Maps well data to the standardized EnzymeML format, yielding time-course data with metadata for further analysis.

## 🔬 Supported Plate Readers

The following table lists currently supported plate reader output formats:

| Manufacturer       | Model                        | File Format |
|--------------------|------------------------------|-------------|
| Agilent            | BioTek Epoch 2               | `xlsx`      |
| Molecular Devices  | SpectraMax 190               | `txt`       |
| Tekan              | Magellan (processing software)| `xlsx`     |
| Tekan              | Spark                        | `xlsx`      |
| Thermo Scientific  | Multiskan SkyHigh            | `xlsx`      |
| Thermo Scientific  | Multiskan Spectrum 1500      | `txt`       |



## 📦 Installation

Install `mtphandler` via PyPI:

```bash
pip install mtphandler # 🚧 not released yet
```
or from source:
```bash
pip install git+https://github.com/FAIRChemistry/MTPHandler.git
```

Please refer to the [documentation](https://fairchemistry.github.io/MTPHandler/) for more information on how to use the package.
