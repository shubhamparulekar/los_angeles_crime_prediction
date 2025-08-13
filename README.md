# Los Angeles Crime Prediction Project

## 📜 Project Overview

This project aims to analyze and predict crime in Los Angeles using the LAPD Crime Dataset (2020-2025). The primary objectives are twofold:
1.  **Crime Type Classification**: To classify crimes into distinct categories ("Assault," "Burglary," "Other") to help in understanding crime patterns.
2.  **Crime Count Forecasting**: To forecast future monthly crime volumes using time series analysis, enabling better resource allocation for law enforcement.

The project leverages **AWS SageMaker** for its computational power and is built using Python with libraries such as Pandas, XGBoost, LightGBM, and Scikit-learn.

---

## ⚙️ Methodology

The project was divided into two main modeling tasks: classification and time series forecasting.

### Crime Type Classification

The goal was to build a model that could accurately classify the type of crime.

* **Data Preparation**:
    * The dataset was loaded from an AWS S3 bucket (`mlc-project-1`).
    * Features were engineered to capture **temporal** (Hour, DayOfWeek, Is_Night), **spatial** (Latitude, Longitude, Distance from Downtown, K-Means Cluster ID), and **demographic** (Victim Age, Sex, Descent) information.
    * Categorical features were one-hot encoded, which expanded the feature space to over 359 dimensions.

* **Dimensionality Reduction**:
    * To manage the high dimensionality and improve model efficiency, **TruncatedSVD** was applied.
    * The feature set was reduced to 100 principal components, which sped up training and reduced the risk of overfitting.

* **Class Balancing**:
    * The dataset was highly imbalanced. To address this, **Synthetic Minority Oversampling Technique (SMOTE)** was used to generate synthetic samples for the minority classes, creating a balanced dataset for training.

* **Model Training**:
    * Three different classification algorithms were trained and evaluated:
        * **Random Forest**: A good baseline model.
        * **LightGBM**: Known for its speed and efficiency.
        * **XGBoost**: A powerful gradient-boosting algorithm capable of handling complex relationships.
    * Hyperparameters like `max_depth`, `n_estimators`, and `learning_rate` were tuned for optimal performance, and early stopping was used in XGBoost to prevent overfitting.

### Crime Count Forecasting

This task focused on predicting the total number of crimes on a monthly basis.

* **Feature Engineering**:
    * Crime data was aggregated by month and year.
    * Key features were created, including **lag features** (crime counts from previous 1, 2, and 3 months) and rolling averages to capture temporal dependencies.

* **Model Training**:
    * The following models were used for forecasting:
        * **Linear Regression**: To establish a simple, interpretable baseline.
        * **Random Forest Regressor**: To capture potential non-linear patterns.
        * **SARIMAX**: A statistical model designed to explicitly handle seasonality and trends.
    * The time series was decomposed into trend, seasonal, and residual components to improve model stability.

---

## 📊 Results and Performance

### Classification Model Results

The models were evaluated based on their accuracy in predicting crime types.

* **XGBoost**: Achieved the highest accuracy of **~83%** and demonstrated a balanced precision and recall across all classes, making it the most reliable model.
<img width="464" height="233" alt="image" src="https://github.com/user-attachments/assets/f729dfa0-d39f-4c1e-b657-e72dab3dd200" />

* **LightGBM**: Performed well with an accuracy of **~74%**, proving to be faster than Random Forest but slightly less accurate than XGBoost.
<img width="456" height="225" alt="image" src="https://github.com/user-attachments/assets/0a966d29-b837-4703-95af-75b59fd4aaf9" />

* **Random Forest**: Served as a baseline with an accuracy of **~70%**, struggling with the more complex and imbalanced patterns in the data.
 <img width="464" height="243" alt="image" src="https://github.com/user-attachments/assets/fe5a8b14-b708-4747-b28a-0b541bef0a83" />


### Forecasting Model Results

The time series models were evaluated on MAE, RMSE, and R² metrics.
* <img width="469" height="154" alt="image" src="https://github.com/user-attachments/assets/a9d38b59-2e28-4564-b24a-af45f7244615" />


* **Linear Regression**: Surprisingly, this simple model provided the **best performance**, outperforming the more complex models. This suggests that the engineered lag and seasonal features captured the underlying linear trends effectively.
<img width="568" height="217" alt="image" src="https://github.com/user-attachments/assets/67a8fe6a-6e9c-49dc-aef4-7090868afdc5" />


* **Random Forest**: Underperformed on this dataset, yielding larger errors.
  <img width="568" height="248" alt="image" src="https://github.com/user-attachments/assets/a7ea3d21-00b6-4288-8ba0-72714fd48eb3" />

* **SARIMAX**: Was highly sensitive to parameter tuning and did not perform as well as the linear model in this evaluation.
<img width="568" height="218" alt="image" src="https://github.com/user-attachments/assets/c1cf31cc-2c56-426e-a78a-4fbb6b8c2f68" />


---

## 💡 Key Takeaways and Future Work

### Strengths

* **Feature Engineering**: The creation of detailed temporal and spatial features was crucial for improving model accuracy, especially for the classification task.
* **XGBoost's Power**: XGBoost proved to be exceptionally effective at handling the class imbalance and complex non-linear relationships in the data.
* **SMOTE Balancing**: Using SMOTE was essential for improving the model's ability to predict rarer crime types, enhancing its overall robustness.

### Limitations

* **Data Noise**: The "Other" crime category is very diverse and may introduce noise that limits model performance.
* **Data Quality**: Missing or inaccurate spatial data for some crime reports posed a challenge.
* **Aggregation Granularity**: Monthly aggregation for the time series model loses finer-grained daily or weekly patterns.


---

## 🚀 How to Run

To reproduce this project, follow these steps:

1.  **Set up Environment**: This project was developed using an **AWS SageMaker Studio Notebook**.
2.  **Data Storage**: Upload the `Crime_Data_from_2020_to_Present.csv` dataset to an S3 bucket named `mlc-project-1`.
3.  **Install Libraries**: Run the first cell in the `MLC_Project_Classification_Model_AWSSagemaker.ipynb` notebook to install all required Python libraries.
    ```python
    !pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn boto3
    ```
4.  **Execute Notebook**: Run the cells in the Jupyter Notebook sequentially to perform data loading, preprocessing, feature engineering, model training, and evaluation.

---

## 👥 Contributors

* Shubham Parulekar
* Yuran Zhang
* Gayatri Mahindrakar
