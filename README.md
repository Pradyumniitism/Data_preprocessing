
# Dataset Preprocessing and Cleaning

This repository contains the preprocessing and cleaning steps performed on the diamond dataset using Python. The steps include handling missing data, outliers, duplicates, and transforming data to prepare it for further analysis.

## Project Overview

This project is focused on the following key data preprocessing and cleaning tasks:

1. **Importing Important Libraries:** 
   - `pandas`, `numpy` for data manipulation.
   - `StandardScaler`, `MinMaxScaler` from `sklearn` for scaling.
   - `matplotlib.pyplot`, `seaborn` for data visualization.

2. **Reading the Dataset:**
   - The dataset is loaded using `pandas.read_csv()` from the local path.

3. **Viewing the Data:**
   - Initial exploration of the data using `.head()`, `.tail()`, `.describe()`, `.info()`, and `.shape()` methods.

4. **Handling Missing Data:**
   - Checked for missing values.
   - Approaches for handling missing data include dropping null values, filling them with specific values (e.g., 0), and filling with the mean of the column.

5. **Handling Outliers:**
   - Identified outliers using the Interquartile Range (IQR) method and Z-Score method.
   - Outliers were either dropped or handled appropriately.

6. **Handling Duplicates:**
   - Identified and removed duplicate rows from the dataset.

7. **Data Transformation:**
   - Converted data types of columns.
   - Replaced specific values based on conditions.
   - Created new columns by combining or manipulating existing columns.
   - Binned numeric data into categories.

8. **Feature Engineering:**
   - Created new features based on conditions or mathematical operations.

9. **Standardization and Normalization:**
   - Applied standardization and normalization techniques to scale the data for further analysis.

## Setup and Installation

To run the preprocessing steps, you need to have Python installed along with the following libraries:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
```

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data Preprocessing and Cleaning Steps

### 1. Importing Important Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Reading the Dataset
```python
diamond = pd.read_csv(r"C:\Users\Pradyumn Sharma\Downloads\kaam\thoda\ml\seaborn\diamonds.csv")
```

### 3. Viewing the Data
```python
diamond.head()
diamond.tail()
diamond.describe()
diamond.info()
diamond.shape
```

### 4. Handling Missing Data
```python
diamond.isnull().sum()
diamond.dropna(inplace=True)
diamond.fillna(0, inplace=True)
diamond['depth'].fillna(diamond['depth'].mean(), inplace=True)
```

### 5. Handling Outliers
- **Using Interquartile Range (IQR) Method:**
```python
q1 = diamond['depth'].quantile(0.25)
q3 = diamond['depth'].quantile(0.75)
iqr = q3 - q1

outliers1 = diamond[(diamond['depth'] < (q1 - 1.5 * iqr)) | (diamond['depth'] > (q3 + 1.5 * iqr))]
```

- **Using Z-Score Method:**
```python
from scipy.stats import zscore

diamond['z_Score'] = zscore(diamond['depth'])
outliers2 = diamond[diamond['z_Score'].abs() > 3]
diamond = diamond[diamond['z_Score'].abs() <= 3]
diamond.drop('z_Score', axis=1, inplace=True)
```

### 6. Handling Duplicates
```python
diamond.duplicated().sum()
diamond.drop_duplicates(inplace=True)
```

### 7. Data Transformation
- **Converting Data Types:**
```python
diamond['depth'] = diamond['depth'].astype('double')
diamond['cut'] = diamond['cut'].astype('category')
```

- **Replacing Values:**
```python
df['column'] = df['column'].replace('old_value', 'new_value')
df.replace({'old_value1': 'new_value1', 'old_value2': 'new_value2'}, inplace=True)
```

### 8. Feature Engineering
- **Creating New Columns:**
```python
diamond['a'] = diamond['x'] + diamond['y']
diamond.drop(columns=['a'], inplace=True)

diamond['depthq'] = diamond['depth'].apply(lambda x: 'Very' if x > diamond['depth'].mean() else 'Okay')
diamond.drop(columns=['depthq'], inplace=True)
```

### 9. Standardization and Normalization
- **Standardization:**
```python
scaler = StandardScaler()
diamond['x1'] = scaler.fit_transform(diamond[['x']])
diamond.drop(columns=['x1'], inplace=True)
```

- **Normalization:**
```python
scalern = MinMaxScaler()
diamond[['x_1', 'depth_1']] = scalern.fit_transform(diamond[['x', 'depth']])
diamond.drop(columns=['x_1', 'depth_1'], inplace=True)
```

## Summary

This preprocessing and cleaning process ensures that the diamond dataset is ready for further analysis, with clean, standardized, and well-structured data.
