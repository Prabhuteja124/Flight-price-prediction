#  Flight Price Prediction

A machine learning project to predict flight ticket prices using supervised regression algorithms. The goal is to help users or businesses estimate airfares accurately based on key flight features like duration, airline, departure/arrival time, and number of stops.

---

##  Dataset Description

The dataset contains historical flight information with the following key features:

- **Airline**: The airline that operates the flight.
- **Source & Destination**: Departure and arrival cities.
- **Date_of_Journey**, **Dep_Time**, **Arrival_Time**: Temporal features.
- **Duration**: Total travel time.
- **Total_Stops**: Number of stops between source and destination.
- **Additional_Info**: Miscellaneous info.
- **Price**: Target variable (flight fare in INR).

---

##  Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Univariate, bivariate, and multivariate visualizations.
   - Correlation heatmaps and distribution plots.

2. **Data Preprocessing**
   - Extracted features from datetime columns.
   - Handled categorical variables using label and one-hot encoding.
   - Applied log transformation to reduce target skewness.
   - Dropped/redundant features and handled outliers.

3. **Model Building**
   - Split data into training and testing sets.
   - Trained multiple regression models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - ElasticNet Regression
     - Decision Tree Regressor
     - Random Forest Regressor
     - Gradient Boosting Regressor
     - XGBoost Regressor
     - LightGBM Regressor
     - Support Vector Regressor (SVR)

4. **Evaluation Metrics**
   - R² Score (Goodness of fit)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)

5. **Hyperparameter Tuning**
   - Used GridSearchCV to fine-tune high-performing models.
   - Compared pre-tuned and post-tuned metrics.

---

##  Outcomes & Insights

- **XGBoost Regressor** consistently outperformed all models before and after tuning with the lowest MAE and highest R². This model is considered the **best candidate for saving and deployment**.
- **Gradient Boosting Regressor** showed the most significant improvement after tuning. It is a strong secondary model and can be saved optionally.
- **Log transformation** of the price target significantly improved the performance of linear models.
- Ridge, Lasso, and ElasticNet performed decently but were outperformed by ensemble-based models.
- Visual comparisons of MAE and R² before and after tuning provided clear insights into model improvement.

---

##  Tools & Technologies Used

- **Programming Language**: Python
- **IDE**: Jupyter Notebook
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn`, `plotly` – Visualization
  - `scikit-learn` – Modeling, tuning, evaluation
  - `xgboost`, `lightgbm` – Ensemble learning
- **Model Tuning**: `GridSearchCV`
- **Version Control**: Git
