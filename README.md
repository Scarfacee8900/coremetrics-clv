# Strategic Customer Retention: Leveraging Cohort Insights for ShopSphere Inc.'s Sustainable Growth

This project focuses on a comprehensive analysis of e-commerce transactional data to uncover customer behavior patterns, evaluate retention lifecycles, and develop a highly targeted customer segmentation framework using RFM metrics and machine learning.

---

## 📌 Executive Summary & Key Project Steps

* **Data Ingestion and Cleaning**: Loaded raw transactional data, performed initial quality checks to confirm zero missing values, and removed duplicate entries to ensure data integrity.
* **Feature Engineering**: Created a dedicated `Revenue` column (Quantity × UnitPrice) and validated that all quantity and price records contained positive values.
* **Monthly Revenue Trend Analysis**: Visualized macro-level revenue trends over time to establish a baseline overview of business performance.
* **Cohort Analysis**: Grouped customers into cohorts based on their first purchase month and generated a retention heatmap to identify drop-off patterns and stabilization rates.
* **RFM Analysis**: Calculated Recency, Frequency, and Monetary metrics for each unique customer to quantify transaction patterns, later scaling these metrics for modeling.
* **K-Means Clustering**: Segmented the standardized RFM data into four distinct groups using K-Means, verifying the optimal cluster count (K) via the Elbow Method, Silhouette Score, and `KElbowVisualizer`.
* **Customer Profiling**: Grouped and labeled the data into actionable consumer tiers to optimize marketing spend and lifetime value.

## ⚠️ Python 3.14+ Environment Patch (Required)

Because Python 3.14 has completely removed the legacy `distutils` module, running this notebook on Python 3.14.x or above will trigger a `ModuleNotFoundError: No module named 'distutils'` due to internal dependencies within the `yellowbrick` visualization engine.

**Fix:** If you are using Python 3.14+, you must create a new code cell at the **absolute top** of the `.ipynb` notebook file, paste the environment patch code below, and execute it before running any machine learning or visualization cells:

```python
import sys
import os

# 1. Force setuptools to enable its internal local distutils backup
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"

# 2. Programmatically map the module hook into the Jupyter kernel's active memory
import setuptools
try:
    import _distutils_hack
    _distutils_hack.do_override()
except ImportError:
    pass

print("Jupyter Kernel environment patched successfully!")
```


---

## 📊 Data Ingestion & Technical Dictionary

### Parsing via Pandas `pd.read_csv()`
The `pd.read_csv()` function serves as the primary gateway for loading delimited transactional data into a usable DataFrame. Its core arguments are categorized below by their operational role:

#### 1. Core Data Location & Format
* **`filepath_or_buffer`**: The local file path string, remote URL, or file-like object.
* **`sep` / `delimiter`**: The character used to split fields (defaults to a comma `,`).
* **`engine`**: The underlying parsing engine (`'c'`, `'python'`, or `'pyarrow'`). The C engine prioritizes execution speed, while Python provides a broader feature set.
* **`encoding`**: Specifier for character encoding protocols (e.g., `'utf-8'` or `'latin-1'`).

#### 2. Column & Index Handling
* **`header`**: Defines which row indices contain column labels (defaults to `0`). Set to `None` if the file lacks headers.
* **`names`**: An explicit array of strings to override or provide column headers.
* **`index_col`**: Identifies specific columns to serve as the row labels (index) of the resulting DataFrame.
* **`usecols`**: A restrictive list of specific columns to load, optimizing RAM footprint by ignoring unwanted data.
* **`dtype`**: A dictionary mapping column names to explicit data types (e.g., `{'ID': str}`).

#### 3. Row Selection & Filtering
* **`skiprows`**: The integer count of rows to skip at the top, or a list of specific target row indices to completely ignore.
* **`skipfooter`**: The integer count of rows to ignore at the bottom of the data file.
* **`nrows`**: The specific maximum number of rows to pull from the start of the file.
* **`skip_blank_lines`**: A Boolean toggle (`True` ignores blank lines; `False` maps them as missing values).

#### 4. Handling Missing & Special Values
* **`na_values`**: Explicit strings to map into `NaN` values during parsing (e.g., `['missing', 'N/A']`).
* **`keep_default_na`**: Boolean choice to append custom `na_values` to the default pandas missing list vs. completely replacing it.
* **`na_filter`**: Disables missing value detection when set to `False` to improve parsing speeds on pre-cleaned files.
* **`true_values` / `false_values`**: Explicit string arrays to map directly into Boolean `True` or `False` types.

#### 5. Date & Time Parsing
* **`parse_dates`**: A Boolean toggle or array list specifying columns to automatically transform into `datetime` objects.
* **`date_format`**: An explicit string structure (e.g., `'%Y-%m-%d'`) passed to accelerate datetime translation.
* **`dayfirst`**: Interprets ambiguous international date strings as day-first (e.g., `10/01/2000` parsed as October 10th if `False`, or January 10th if `True`).

#### 6. Memory & Performance
* **`chunksize`**: Returns an iterable file-stream object chunked by a specific row count, preventing RAM exhaustion on large datasets.
* **`low_memory`**: Internally chunks data during ingest to save RAM (enabled by default, but can trigger mixed-type warnings).
* **`memory_map`**: Maps the file handle directly to memory storage on supported operating systems for faster access.

#### 7. Error & Quoting Control
* **`on_bad_lines`**: Sets behavior when lines contain unexpected field counts (`'error'`, `'warn'`, or `'skip'`).
* **`quotechar` / `quoting`**: Configures the wrapper characters for embedded text and how the parser unwraps them.
* **`comment`**: A designated single-character indicator (e.g., `#`) telling the parser to ignore any trailing text on that line.

### Data Dictionary
The underlying dataset consists of structural transaction details outlined below:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **InvoiceNo** | String | A unique 6-digit integral number assigned to each transaction. |
| **StockCode** | String | A 5-digit integral number uniquely identifying each product. |
| **Description** | String | Product name/description text. |
| **Quantity** | Integer | The transaction volume quantities per product. |
| **InvoiceDate** | DateTime | The specific date and time when the transaction record was generated. |
| **UnitPrice** | Float | The individual item unit price in British Pounds (GBP). |
| **CustomerID** | String | A unique 5-digit integral identifier assigned to individual clients. |
| **Country** | String | The residency location of the transacting customer. |

---

## 🔍 Data Exploratory Goals & Methodologies

Before applying modeling transformations, exploring the raw dataset's characteristics achieves several key diagnostic targets:
* **Assessing Quality**: Identifies outliers, formatting inconsistencies, or corrupted records that distort models.
* **Evaluating Cardinality**: Explores the unique distribution ranges within categorical fields (e.g., measuring unique country distributions).
* **Data Cleansing**: Highlights zero-variance features (columns showing identical values across all rows) to prune them out before machine learning.
* **Feature Target Verification**: Compares the row count totals against `nunique()` outputs to check for primary key viability.

---

## 📈 Cohort Analysis Deep Dive

### 1. Conceptual Framework
Cohort analysis organizes users into tracking buckets based on a shared milestone within a specific timeframe. Instead of viewing aggregate performance metrics, it monitors how behavior evolves over an isolated customer lifecycle. For this project, cohorts are built based on the month of a customer's first purchase.

### 2. Implementation Pipeline

(Identify First Purchase Month) ➡️ (Calculate Cohort Index) ➡️ (Count Unique Customers) ➡️ (Pivot Matrix) ➡️ (Compute Retention Rate %)


1. **Defining Cohorts**: Grouped data by `CustomerID` and isolated the minimum `InvoiceMonth` value.
   $$\[\text{Cohort Month} = \min(\text{InvoiceMonth})_{\text{CustomerID}}\]$$
2. **Calculating Cohort Index**: Calculated the elapsed monthly steps between a transaction date and the customer's baseline Cohort Month:
   $$\text{Cohort Index} = (\text{Year}_{\text{diff}} \times 12) + \text{Month}_{\text{diff}} + 1$$
   *An index of 1 signifies that the transaction occurred within the exact same month as the initial acquisition.*
3. **Aggregating Counts**: Structured the unique `CustomerID` counts across overlapping combinations of Cohort Months and Cohort Indexes.
4. **Pivoting Structure**: Transformed the long table structure into a wide layout where the rows represent specific Cohort Months and columns represent sequential Cohort Indexes.
5. **Calculating Retention Rates**: Divided the customer count of an active index cell by the starting population baseline of that cohort (Index 1):
   \[\text{Retention Rate} = \frac{\text{Cohort Pivot Value}_{(\text{Index } n)}}{\text{Cohort Size}_{(\text{Index 1})}}\]
6. **Heatmap Generation**: Rendered standard percentage values using `sns.heatmap(retention_matrix, annot=True, cmap='YlGnBu', fmt='.0%')`.

### 3. Key Observations & Recommendations
* **The Onboarding Drop-Off**: Most cohorts showed an immediate drop between Index 1 and Index 2. 
  * *Strategy*: Deploy focused onboarding paths, welcome incentives, and high-touch automated workflows during the first 30 days to build long-term retention habits.
* **Long-Term Stabilization**: Following the initial churn drop, customer retention rates flattened and decayed gradually across remaining periods.
* **High-Performing Exceptions**: The `2023-12` cohort showed exceptional retention, holding at 40% even at Index 12.
  * *Strategy*: Deconstruct historical performance variables during December 2023 (such as special marketing promotions, item launches, or seasonal source tracking) to isolate and replicate those high-performing acquisition channels.
* **Re-engagement Workflows**: Dropping curves indicate diminishing customer interaction.
  * *Strategy*: Set up automated win-back triggers offering personalized discount structures and recommendations to pull slipping cohorts back into the purchase funnel.

---

## 💎 RFM Analysis & Feature Engineering

### 1. The Core Metrics
RFM quantifies past purchasing history to score behavioral trends:
* **Recency (R)**: Days elapsed since a customer's last purchase. Lower numbers indicate higher recent engagement.
* **Frequency (F)**: Total count of unique orders. Higher figures display brand relationship and loyalty.
* **Monetary (M)**: Total financial spending. Identifies high-value spenders vs. low-margin purchasers.

### 2. Engineering Snapshot Recency Values
To compute recency without real-time data decay, the workflow establishes a data-driven reference date exactly 24 hours after the maximum timestamp in the source dataset:
```python
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
```
* **Why add a day?** This configuration ensures that customers transacting at the final recorded second of the dataset default to a Recency of 1 rather than 0, preventing division or formatting calculation errors in downstream scaling models.

### 3. The Code Engine Breakdown
The pipeline aggregates transactions into distinct profiles using a single, optimized expression:
```python
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Revenue': 'sum'
}).reset_index()
```
* **`lambda x: (reference_date - x.max()).days`**: Isolates the customer's latest purchase timestamp, subtracts it from the dataset's snapshot reference date, and returns an integer day count.
* **`'InvoiceNo': 'nunique'`**: Counts unique invoice identification numbers to track unique visits while ignoring raw line-item counts.
* **`'Revenue': 'sum'`**: Tallies every currency unit spent by the customer to calculate their total financial contribution.
* **`.reset_index()`**: Flattens the output index structure, shifting `CustomerID` from a row label back into a spreadsheet-style data column for modeling compatibility.

### 4. Preparation for Clustering
* **Distribution Properties**: Initial tests showed skewed monetary and frequency metrics, signaling high variance between occasional shoppers and high-value buyers.
* **Standard Scaling**: Because Recency, Frequency, and Monetary scores use different units and numeric scales, they were normalized using `StandardScaler` from `sklearn.preprocessing`. This centers values to a mean of 0 and a standard deviation of 1, preventing the large scale of monetary fields from dominating the clustering model.

---

## 🤖 Cluster Optimization & Profiling

### 1. Determining K (Optimal Clusters)
* **Elbow Method**: Monitors the reduction of inertia (sum of squared distances to cluster centers) across increasing K steps. The rate of decay slowed down smoothly without a sharp bend, suggesting a target of 3 or 4 viable partitions.
* **Silhouette Analysis**: Evaluates structural separation by measuring how closely items match their own cluster compared to neighboring options (scale from -1 to +1). Peak average scores across a range of K=2 to K=10 highlight the ideal cluster choice.
* **`KElbowVisualizer`**: Automated tool that maps distortion scores to pinpoint the optimal balance between cluster compactness and computational complexity.

### 2. Cluster Profiles & Segment Playbooks

Evaluating the final four K-Means groupings reveals distinct customer segments with specific marketing actions:

(Cluster 3: Champions) 👑  --> Max Value, High Frequency(Cluster 0: Loyals)    ❤️  --> Consistent Value & Cadence(Cluster 1: New/Occ.)  🌱  --> High Volume, Low Individual Value(Cluster 2: At-Risk)   ⚠️  --> High Recency, Slipping Retention

#### 🛒 Cluster 0: Loyal Customers
* **Profile Metrics**: N = 1,367 customers. Moderate recency (≈ 47.1 days since last buy), high frequency (≈ 6.3 unique transactions), and substantial monetary contribution (\$269,730.0 aggregate spend).
* **Strategic Playbook**: Provide loyalty rewards, send personalized repeat purchase incentives, and run targeted cross-sell campaigns to expand their product category mix.

#### 🌱 Cluster 1: New / Occasional Customers
* **Profile Metrics**: N = 1,733 customers (largest segment). Moderate recency (≈ 45.0 days), but low purchasing frequency (≈ 2.4 transactions) and lower overall financial spending (\$99,169.9 total spend).
* **Strategic Playbook**: Use engagement tracks to boost purchase velocity and basket sizes. Deliver automated onboarding guides for newer accounts, and send friendly shopping reminders to irregular buyers.

#### ⚠️ Cluster 2: At-Risk / Churned Customers
* **Profile Metrics**: N = 719 customers. Highly problematic recency windows (≈ 186.8 days since last transaction). They show moderate historical frequency (≈ 2.9 transactions) and total spending values (\$125,700.8), but have stalled out over time.
* **Strategic Playbook**: Launch targeted win-back campaigns. Use aggressive win-back discounts, custom account credits, or personalized "we miss you" product recommendations before they churn permanently.

#### 👑 Cluster 3: Champions / VIPs
* **Profile Metrics**: N = 553 customers (highest-value segment). Outstanding recency (≈ 41.5 days), very high frequency (≈ 11.2 distinct orders), and the highest overall monetary contribution (\$490,713.6 aggregate value).
* **Strategic Playbook**: Provide exclusive VIP perks, dedicated support tiering, first-look product drops, and direct feedback loops. Nurture this group as active brand advocates to drive word-of-mouth growth.
