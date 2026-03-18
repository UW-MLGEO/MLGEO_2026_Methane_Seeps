# Predicting Chemical Concentrations at Southern Hydrate Ridge Methane Seeps

## Overview

This project develops machine learning models to predict key dissolved gas concentrations (primarily methane and hydrogen) at the Southern Hydrate Ridge methane seep site using multimodal geophysical data from the Ocean Observatories Initiative (OOI) Regional Cabled Array (RCA). By mapping robust physical measurements (seismometers, pressure sensors, and an acoustic Doppler current profiler) to residual gas analyzer–derived concentrations, we aim to fill instrument downtime and data gaps in seep chemistry. This README provides a concise technical overview; see `docs/writeup.md` for the full narrative description and figures.

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
- Mean amplitude
- Spectral features: top dominant frequencies and their power
- Average frequency and amplitude across all components
- Total: 11

**Acoustic Features**:
- Mean velocity in 22-second window
- Maximum velocity
- Minimum velocity
- Median velocity

**Model Architecture**:
- Algorithm: Random Forest Regressor (scikit-learn)
- Configuration:
  - 800 trees
  - Max depth: 12
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
- Validation: 5-fold cross-validation

**Advantages**:
- Captures temporal patterns and phase relationships
- Learns hierarchical features automatically
- Better for sequential dependencies
- No manual feature engineering required

## Model Evaluation

### Data Split Strategy

- **Training set**: 70% of data (further split in 5-fold CV)
- **Validation sets**: 15% of data, 5 folds for cross-validation
- **Test set**: 15% held-out data for final evaluation

### Performance Metrics

- **RMSE** (Root Mean Squared Error): Prediction accuracy
- **R²** (Coefficient of Determination): Explained variance
- **MAE** (Mean Absolute Error): Average prediction error

## Results Summary

### Random Forest Performance

- Cross-validation R²: 90%
- Test set R²: ~90%
- Test set residual: ~85%

**Top Feature Importances**:
- Dominant spectral power from seismic East component (E_power1) - 39%
- Dominant spectral frequency from seismic East component (E_freq1) - 10%
- Average amplitude from seismic East component (E_mean) - 9%
- Median bubble plume velocity from acoustic velocity profiler (adcp_median) - 9%
- Mean bubble plume velocity from acoustic velocity profiler (adcp_mean) - 9%

### Model Comparison

The CNN model resulted in poor performance on raw time series data, while the RF showed robust performance. We opted to use the RF model in our final analysis.

## Repository Structure

```
MLGEO_2026_Methane_Seeps/
├── notebooks/                          # Analysis and modeling notebooks
│   ├── random_forest_features.ipynb    # Feature-based RF model
│   ├── time_series_cnn.ipynb           # CNN on raw time series
│   ├── shr_seismicity*.ipynb           # Seismic and short-duration event exploration
│   ├── MSDataToCSV.py                  # MASSPA data-to-CSV utilities
│   └── ...                             # Additional EDA and figure notebooks
├── figures/                            # Exported figures for writeup/docs
├── docs/
│   ├── README.md                       # This file (project overview)
│   ├── writeup.md                      # Full scientific writeup
│   ├── environment.yml                 # Conda environment
│   └── requirements.txt                # pip requirements
│   ├── MSDataCollectorV1/              # Raw data collection utilities
│   ├── pyproject.toml                  # Python project metadata
│   └── uv.lock                         # Locked dependency versions


## Dependencies

This project’s environment is defined in `docs/environment.yml` (Conda) and `docs/requirements.txt` (pip). The `pyproject.toml` in `MLGEO_2026_Methane_Seeps/docs/` provides a lightweight dependency specification for the subproject.

### Recommended Python

- Python >= 3.8

### Core Packages

- Scientific computing: `numpy`, `pandas`, `scipy`
- Seismic and NetCDF I/O: `obspy`, `netCDF4`, `cftime`, `xarray`
- Machine learning: `scikit-learn`, `torch`/`pytorch`, `torchvision`
- Visualization: `matplotlib`, `seaborn`
- Notebooks and tooling: `jupyter`, `ipykernel`, `notebook`, `jupyterlab`, `tqdm`, `statsmodels`
- Web and utilities: `requests`, `bs4`, `datetime`

To reproduce the environment with Conda:

```bash
conda env create -f docs/environment.yml
conda activate mlgeo-methane-seeps
```

Or with pip (inside a virtual environment):

```bash
pip install -r docs/requirements.txt
```

### External Data Services

- **IRIS FDSN Web Services**: Seismic waveform data
   - Client: `obspy.clients.fdsn.Client("IRIS")`
   - Networks: OO (Ocean Observatories Initiative)

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
Isaac Olson

## Acknowledgments

- **Ocean Observatories Initiative (OOI)**: Data infrastructure and access
- **EarthScope Consortium**: Seismic waveform services
- **University of Washington Earth and Space Sciences**: Computational resources
- **Dr. Akshay Mehra and Henry Yuan**: Guidance and project feedback

## References

### Scientific Background

MacLeod, L. M. F., & Wilcock, W. S. D. (2025). Nonseismic short-duration events offshore Cascadia: Characteristics and potential origin. Seismological Research Letters, 96(2A), 706–720. [https://doi.org/10.1785/0220240367](https://doi.org/10.1785/0220240367)

Marcon, Y., Kelley, D. S., Thornton, B., Manalang, D., & Bohrmann, G. (2021). Variability of natural methane bubble release at Southern Hydrate Ridge. Geochemistry, Geophysics, Geosystems, 22, e2021GC009894. [https://doi.org/10.1029/2021GC009894](https://doi.org/10.1029/2021GC009894)

Philip, B. T., A. R.Denny, E. A.Solomon, and D. S. Kelley (2016). Time-series measurements of bubble plume variability and water column methane distribution above Southern Hydrate Ridge, Oregon, Geochem. Geophys. Geosyst., 17, 1182–1196, doi:10.1002/2016GC006250.

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
