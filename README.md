## Predictive Modeling of Permeate Flux in Vacuum Membrane Distillation: A Performance Comparison

### Project Overview
This project evaluates the efficacy of classical statistical methods against data-driven Machine Learning algorithms for predicting permeate flux in Vacuum Membrane Distillation (VMD) using a limited experimental dataset (n=31). Based on a Central Composite Design (CCD), 4 input parameters (Feed Temperature, Vacuum Pressure, Feed Flow Rate, and Feed Concentration) were utilized to predict the primary output parameter (Permeate Flux). To account for data sparsity, Leave-One-Out Cross-Validation (LOOCV) is utilized. The repository compares four predictive architectures: a quadratic Response Surface Methodology (RSM) model, a Multilayer Perceptron (MLP/ANN), a Support Vector Regressor (SVR), and a Random Forest (RF).

### Repository Structure
* **`data/`**
  * `VMD_Data.csv`: The 31-row experimental dataset generated via Central Composite Design.
* **`models/`**
  * `VMD_RSM.ipynb`: Jupyter notebook detailing the Ordinary Least Squares (OLS) regression for the quadratic RSM model.
  * `VMD_MLP.py`: Python script to train and evaluate the Neural Network model.
  * `VMD_RF.py`: Python script to train and evaluate the Random Forest model.
  * `VMD_SVR.py`: Python script to train and evaluate the Support Vector Regressor.
* `README.md`

### Prerequisites and Dependencies
* Python 3.x
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `statsmodels`

### Installation and Execution
1. Clone or download this repository to a local directory.
2. Install the required dependencies: `pip install scikit-learn pandas numpy matplotlib statsmodels`
3. **RSM Modeling:** Open and run all cells in `VMD_RSM.ipynb` to view the statistical optimization, regression coefficients, and parity plots.
4. **Machine Learning Models:** Execute the individual Python scripts directly from the root directory (e.g., `python VMD_RF.py`). 
5. Ensure each script is configured to load data correctly from the `data/` folder (e.g., `pd.read_csv('data/VMD_Data.csv')`).

### Key Results
* **Response Surface Methodology (RSM):** Best overall performer ($R^2 = 0.995$, $Q^2 = 0.978$). Effectively captured the thermodynamic gradients without overfitting the sparse data.
* **Artificial Neural Network (MLP):** Second best performer ($R^2 = 0.928$). Utilized a 'tanh' activation function to handle non-linearities but struggled slightly to generalize smooth thermodynamic gradients across the entire domain.
* **Support Vector Regressor (SVR):** Moderate performance ($R^2 = 0.890$). Lack of dense data clusters constrained the support vectors, leading to sub-optimal boundary definitions.
* **Random Forest (RF):** Worst performing model ($R^2 = 0.856$). Failed to capture the continuous, curvilinear nature of the vapor pressure driving force due to data sparsity.
