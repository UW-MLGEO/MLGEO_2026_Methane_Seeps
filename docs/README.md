# Predicting Chemical Concentrations at Southern Hydrate Ridge Methane Seeps

## Overview

This project develops machine learning models to predict key chemical concentrations (methane and hydrogen) at the Southern Hydrate Ridge methane seep site using multimodal geophysical data from the Ocean Observatories Initiative (OOI) Regional Cabled Array (RCA). By leveraging reliable, long-lasting instruments (seismometers, pressure sensors, acoustic Doppler current profilers), we aim to predict biogeochemically important chemical concentrations during time periods when direct mass spectrometer measurements are unavailable.

## Scientific Context

Southern Hydrate Ridge, located offshore Oregon, is a methane seep site of significant biogeochemical importance. The site features active methane venting, gas hydrate deposits, and complex fluid flow dynamics. Understanding temporal variations in methane (CH₄) and hydrogen (H₂) concentrations is crucial for:

- Quantifying methane flux to the ocean and atmosphere
- Understanding subsurface biogeochemical processes
- Characterizing seep dynamics and episodic venting events
- Assessing climate-relevant greenhouse gas emissions

However, mass spectrometer instruments require frequent maintenance and calibration, leading to data gaps. This project addresses these gaps by training models on reliable geophysical proxies.

## Data Sources

### Ocean Observatories Initiative (OOI) Regional Cabled Array

All data are collected from the OOI RCA infrastructure at Southern Hydrate Ridge:

#### Input Features (Predictors):

1. **Seismic Data**
   - Station: OO.HYS14
   - Channels: HHZ, HHN, HHE (200 Hz sampling rate)
   - Three-component broadband seismometer data
   - Sensitive to ground motion, tremor, and fluid flow dynamics

2. **Acoustic Data**
   - Instrument: ADCP (Acoustic Doppler Current Profiler)
   - Dataset: RS01SUM2-MJ01B-12-ADCPSK101
   - Measures bubble plume velocity and water column dynamics
   - Hourly measurements of current profiles

3. **Pressure Data**
   - Bottom pressure recorders (BPR)
   - Sensitive to tidal loading and seasonality trends
   - Used to detrend average bubble plume velocity

#### Target Variables (Labels):

4. **Residual Gas Analyzer Data**
   - Methane (CH₄) concentration (nM/L)
   - Hydrogen (H₂) concentration (nM/L)
   - Hydrogen Sulfide (H₂S) concentration (mM/L)
   - Nitrogen (N₂) concentration (nM/L)
   - Oxygen (O₂) concentration (nM/L)
   - Carbon dioxide (CO₂) concentration (nM/L)
   - Derived from partial pressure measurements using Henry's Law
   - Ex: `methane_concentration_2017.csv`

## Data Processing Pipeline

### Temporal Alignment

All seismic are aligned to **22-second time windows** corresponding to RGA measurement timestamps, while ADCP measurements are on an hour long basis.

### Seismic Data Processing

For each 22-second window:

1. **Download**: Query IRIS FDSN web services for three-component data
2. **Detrend**: Remove linear trends using ObsPy
3. **Taper**: Apply 5% Hanning window to edges
4. **Highpass Filter**: Highpass filter at 2 Hz cutoff
   - Removes microseism noise and long-period ocean loading
   - Focuses on signals related to fluid flow and bubble dynamics
5. **Quality Control**: Check for gaps, NaN values, and data quality

### Acoustic (ADCP) Data Processing

1. **Load NetCDF**: Read hourly ADCP velocity data
2. **Temporal Extraction**: For each 22-second window, extract all ADCP measurements - may replicate many measurements across windows due to hourly binning constraint
3. **Bin Averaging**: If multiple depth bins present, compute mean across bins
4. **Unit Conversion**: Convert m/s to cm/s where appropriate

### Target Variable Calculation

All chmeical concentrations are calculated from mass spectrometer partial pressure measurements using **Henry's Law**:

```
C = K_H × P_partial
```

Where:
- C = dissolved gas concentration (nM/L)
- K_H = Henry's Law constant (temperature and depth dependent - unique for each concentration, courtesy of NOAA)
- P_partial = partial pressure measured by mass spectrometer

## Machine Learning Approaches

This project initially implemented **two complementary modeling strategies**, each suited to different aspects of the data:

### Approach 1: Feature-Based Random Forest Regression

**Training Data Format**: Extracted statistical and spectral features

**Feature Extraction** (per 22-second window):

**Seismic Features** (per component: Z, N, E):
- Amplitude statistics: max, min, mean, median
- Spectral features: top 3 dominant frequencies and their power
- Total: ~24 features per window

**Acoustic Features**:
- Mean velocity in 22-second window
- Maximum velocity
- Minimum velocity
- Standard deviation

**Model Architecture**:
- Algorithm: Random Forest Regressor (scikit-learn)
- Configuration:
  - 400 trees
  - Max depth: 12
  - Min samples split: 5
  - Min samples leaf: 2
  - Max features: 'sqrt'
- Validation: 5-fold cross-validation

**Advantages**:
- Interpretable feature importances
- Handles non-linear relationships
- Robust to outliers
- Fast training and inference

### Approach 2: Time Series Convolutional Neural Network

**Training Data Format**: Raw time series arrays (post-filtering)

**Data Structure** (per 22-second window):
- Seismic: 3 channels × 4400 samples (200 Hz × 22 sec)
- Acoustic: Variable samples (hourly measurements interpolated)

**Model Architecture**:
- Framework: PyTorch
- Architecture: 1D CNN Regressor
  - Input layer: Reshape features to (batch, 1, n_features)
  - Convolutional blocks:
    - Conv1D: 1 → 64 channels, kernel=3
    - BatchNorm + ReLU + Dropout(0.2)
    - Conv1D: 64 → 128 channels, kernel=3
    - BatchNorm + ReLU + Dropout(0.2)
    - Conv1D: 128 → 64 channels, kernel=3
    - BatchNorm + ReLU + Dropout(0.2)
  - Global Average Pooling
  - Fully connected layers: 64 → 128 → 64 → 32 → 1
  - Dropout(0.3) in FC layers
- Training:
  - Optimizer: Adam (lr=0.001)
  - Loss: MSE (Mean Squared Error)
  - Epochs: 100 with early stopping (patience=15)
  - Batch size: 32
- Validation: 5-fold cross-validation

**Advantages**:
- Captures temporal patterns and phase relationships
- Learns hierarchical features automatically
- Better for sequential dependencies
- No manual feature engineering required

## Model Evaluation

### Data Split Strategy

- **Training set**: 70% of data (further split in 5-fold CV)
- **Validation sets**: 5 folds for cross-validation
- **Test set**: 15% held-out data for final evaluation

### Performance Metrics

- **RMSE** (Root Mean Squared Error): Prediction accuracy
- **R²** (Coefficient of Determination): Explained variance
- **MAE** (Mean Absolute Error): Average prediction error

## Results Summary

### Random Forest Performance

- Cross-validation R²: 90%
- Test set R²: 90%
- Test set residual: 

**Top Feature Importances**:
- Dominant spectral power from seismic East component (E_power1) - 35%
- Dominant spectral frequency from seismic East component (E_freq1) - 10%
- Average amplitude from seismic East component (E_mean) - 9%
- Median bubble plume velocity from acoustic velocity profiler (adcp_median) - 9%
- Mean bubble plume velocity from acoustic velocity profiler (adcp_mean) - 9%

### Model Comparison

The CNN model resulted in poor performance on raw time series data, while the RF showed robust performance. We opted to use the RF model in our final analysis.

## Repository Structure

```
mlgeo-methane-seeps/
├── shr_seismicity_relevant_dates.ipynb          # Feature-based RF analysis
├── shr_seismicity_relevant_dates_time_series.ipynb  # Time series CNN analysis
├── data/
│   ├── methane_concentration_2017.csv           # Target variables
│   ├── RS01SUM2-MJ01B-12-ADCPSK101_*.nc        # ADCP data
│   └── seismic_features_*.csv                   # Extracted features
├── models/
│   ├── rf_model_final.pkl                       # Trained Random Forest
│   ├── cnn_model_final.pth                      # Trained CNN
│   └── transformers.pkl                         # Data transformers
├── results/
│   ├── correlation_matrix_*.png
│   ├── pca_analysis_*.png
│   ├── random_forest_5fold_cv_results.png
│   └── cnn_regression_5fold_cv_results.png
└── MLGEO_2026_Hydrothermal_Vents/
    └── docs/
        └── README.md                             # This file
```

## Dependencies

### Python Environment

```python
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Seismic data processing
obspy>=1.3.0

# NetCDF file handling
netCDF4>=1.5.7

# Machine learning
scikit-learn>=1.0.0
torch>=1.10.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Data access
requests>=2.26.0
```

### External Data Services

- **IRIS FDSN Web Services**: Seismic waveform data
  - Client: `obspy.clients.fdsn.Client("IRIS")`
  - Networks: OO (Ocean Observatories Initiative)

## Usage

### 1. Feature-Based Random Forest Workflow

```python
# Run feature extraction and RF training
jupyter notebook shr_seismicity_relevant_dates.ipynb

# Key cells:
# - Data loading and window creation
# - Feature extraction from seismic/acoustic data
# - Random Forest training with 5-fold CV
# - Model evaluation and visualization
```

### 2. Time Series CNN Workflow

```python
# Run time series extraction and CNN training
jupyter notebook shr_seismicity_relevant_dates_time_series.ipynb

# Key cells:
# - Time series extraction (22-sec windows)
# - Data storage as pickle files
# - CNN model training with PyTorch
# - Model comparison with Random Forest
```

### 3. Loading Pre-trained Models

```python
# Load Random Forest
import pickle
with open('models/rf_model_final.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load CNN
import torch
checkpoint = torch.load('models/cnn_model_final.pth')
cnn_model = CNN1DRegressor(input_dim=n_features)
cnn_model.load_state_dict(checkpoint['model_state_dict'])
```

## Scientific Applications

### Methane Flux Estimation

- Continuous prediction of CH₄ concentrations during instrument downtime
- Improved temporal resolution for flux calculations
- Better characterization of episodic venting events

### Seep Dynamics Studies

- Correlation between seismic tremor and chemical release
- Relationship between bottom currents and plume dispersal
- Pressure-driven fluid migration patterns

### Early Warning Systems

- Real-time prediction of elevated methane concentrations
- Detection of anomalous venting events
- Integration with autonomous sampling strategies

## Limitations and Future Work

### Current Limitations

1. **Temporal Coverage**: Training data limited to 2017 when all instruments operational
2. **Spatial Coverage**: Single seismic station (HYS14)
3. **ADCP Sampling**: Hourly measurements limit temporal resolution
4. **Seasonal Bias**: May not capture full range of environmental conditions

### Future Directions

1. **Multi-year Training**: Incorporate data from 2014-2020
2. **Multi-station Ensemble**: Use all HYS1* stations, even at 8Hz sampling, to expand coverage for low-frequency events.
3. **Additional Features**: 
   - Temperature and salinity from CTD sensors
   - Tidal phase and magnitude
4. **Transfer Learning**: Apply models to other seep sites
5. **Real-time Deployment**: Implement models in OOI cyberinfrastructure
6. **Physics-informed ML**: Incorporate fluid dynamics constraints

## Contributors

Christina Stuhl
David Lovett
Michael Hemmett

## Acknowledgments

- **Ocean Observatories Initiative (OOI)**: Data infrastructure and access
- **EarthScope Consortium**: Seismic waveform services
- **University of Washington Earth and Space Sciences**: Computational resources
- **Dr. Akshay Mehra and Henry Yuan**: Guidance and project feedback

## References

### Scientific Background

MacLeod, L. M. F., & Wilcock, W. S. D. (2025). Nonseismic short-duration events offshore Cascadia: Characteristics and potential origin. Seismological Research Letters, 96(2A), 706–720. [https://doi.org/10.1785/0220240367](https://doi.org/10.1785/0220240367)

Marcon, Y., Kelley, D. S., Thornton, B., Manalang, D., & Bohrmann, G. (2021). Variability of natural methane bubble release at Southern Hydrate Ridge. Geochemistry, Geophysics, Geosystems, 22, e2021GC009894. [https://doi.org/10.1029/2021GC009894](https://doi.org/10.1029/2021GC009894)

Reeburgh, W. S. (2007). Oceanic methane biogeochemistry. Chemical Reviews, 107(2), 486–513. [https://doi.org/10.1021/cr050362v](https://doi.org/10.1021/cr050362v)

Römer, M., Sahling, H., Pape, T., Bahr, A., Feseker, T., Wintersteller, P., & Bohrmann, G. (2014). Microbial abundance and diversity patterns associated with sediments and carbonates from methane seep environments of Hydrate Ridge, Oregon. Frontiers in Marine Science, 1, Article 44. [https://doi.org/10.3389/fmars.2014.00044](https://doi.org/10.3389/fmars.2014.00044)

Sahling, H., Galkin, S. V., Salyuk, A., Greinert, J., Foerstel, H., Piepenburg, D., & Suess, E. (2002). Macrofaunal community structure and sulfide flux at gas hydrate deposits from the Cascadia convergent margin, NE Pacific. Marine Ecology Progress Series, 231, 121–138.

Treude, T., Boetius, A., Knittel, K., Wallmann, K., & Jørgensen, B. B. (2003). Anaerobic oxidation of methane above gas hydrates at Hydrate Ridge, NE Pacific Ocean. Marine Ecology Progress Series, 264, 1–14.

Tryon, M. D., Brown, K. M., & Torres, M. E. (2001). Complex flow patterns through Hydrate Ridge and their impact on seep biota. Geophysical Research Letters, 28(15), 2863–2866. [https://doi.org/10.1029/2000GL012566](https://doi.org/10.1029/2000GL012566)

Luff, R., Wallmann, K., Aloisi, G., et al. (2008). Miniaturized biosignature analysis reveals implications for the formation of cold seep carbonates at Hydrate Ridge (off Oregon, USA). Biogeosciences, 5, 731–741.


### OOI Documentation

Ocean Observatories Initiative. "Regional Cabled Array." https://oceanobservatories.org/

## License

MIT open license

---

**Last Updated**: March 2026  
**Project Status**: Not currently under development
