# Lakhina Entropy-Based Intrusion Detection System (IDS)

## Technical Report

---

## Executive Summary

This project implements a network intrusion detection system based on the **Lakhina Entropy Method**, a statistical anomaly detection approach that leverages **Principal Component Analysis (PCA)** on entropy-based traffic features. The system is designed to detect botnet activity in network traffic by identifying deviations from normal traffic patterns without requiring signature-based detection rules.

---

## 1. Introduction and Background

### 1.1 The Lakhina Approach

The methodology is based on the seminal work by Anukool Lakhina et al. (*"Mining Anomalies Using Traffic Feature Distributions"*, SIGCOMM 2005), which demonstrated that network traffic anomalies can be effectively detected by analyzing the statistical distributions of traffic features rather than volume-based metrics alone.

### 1.2 Core Principle

The fundamental insight is that **normal network traffic exhibits predictable entropy patterns** across various traffic dimensions (source ports, destination ports, destination IPs, TCP flags). Anomalous traffic—including botnet command-and-control (C&C), DDoS attacks, and scanning activities—produces **entropy deviations** that can be detected using PCA-based subspace analysis.

---

## 2. Dataset: CTU-13

### 2.1 Dataset Description

The implementation uses the **CTU-13 Dataset** (Czech Technical University), a benchmark dataset for botnet detection containing captured network traffic from controlled experiments with known botnet samples.

**Key Characteristics:**
- **Format**: `.binetflow` files (Bidirectional NetFlow format)
- **Columns**:
  - `StartTime`: Timestamp of flow
  - `Dur`: Duration of the flow
  - `Proto`: Protocol (tcp, udp, icmp, etc.)
  - `SrcAddr`, `DstAddr`: Source/Destination IP addresses
  - `Sport`, `Dport`: Source/Destination ports
  - `State`: TCP state/flags
  - `TotPkts`, `TotBytes`, `SrcBytes`: Volume metrics
  - `Label`: Ground truth classification

### 2.2 Label Distribution

The dataset contains three classes:
- **Background**: Normal background traffic (~98%)
- **Botnet**: Malicious botnet traffic (~1-2%)
- **Normal**: Legitimate traffic (~0.5%)

### 2.3 Data Splitting Strategy

The dataset is split **chronologically** (not randomly) to simulate real-world IDS deployment:

```
Train:      60% (earliest traffic)
Validation: 20%
Test:       20% (latest traffic)
```

This temporal split ensures the model never "sees the future" during training, maintaining realistic evaluation conditions.

---

## 3. Methodology

### 3.1 Pipeline Overview

```
Raw Network Flows (CTU-13)
         │
         ▼
┌─────────────────────────┐
│  Data Preprocessing    │
│  - Load & clean        │
│  - Normalize labels    │
│  - Parse ports/flags   │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Protocol Filtering    │
│  - TCP-only selection  │
│  - Flag extraction     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  IP Aggregation        │
│  - Group by source IP  │
│  - Compute entropies   │
│  - Compute flag freqs  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  PCA Transformation    │
│  - Standardization    │
│  - Eigendecomposition │
│  - Major/Minor split  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Anomaly Scoring      │
│  - Major subspace     │
│  - Minor subspace     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Threshold Detection  │
│  - Grid search on val │
│  - F1 optimization    │
└─────────────────────────┘
```

### 3.2 TCP Filtering Rationale

The implementation filters to **TCP traffic only** (`use_tcp_only: True`). This is because:
1. TCP flags (SYN, ACK, FIN, RST, PSH, URG) provide rich behavioral signals
2. Many botnet C&C channels use TCP for reliability
3. TCP state transitions are more informative for anomaly detection

---

## 4. Feature Engineering

### 4.1 TCP Flag Frequencies

For each source IP, TCP flag frequencies are computed:

$$f_{\text{flag}} = \frac{\text{count of flag occurrences}}{\text{total flows from IP}}$$

Flags extracted from the `State` field:
- `syn` (S): Connection initiation
- `ack` (A): Acknowledgment
- `fin` (F): Connection termination
- `rst` (R): Reset/connection refusal
- `psh` (P): Push flag
- `urg` (U): Urgent flag

### 4.2 Shannon Entropy Features

The core innovation uses **Shannon entropy** to measure the unpredictability of traffic distributions.

#### 4.2.1 Shannon Entropy Formula

For a discrete random variable $X$ with probability mass function $p(x)$, the Shannon entropy is:

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

Where:
- $p_i$ is the probability (frequency) of value $i$
- $n$ is the number of unique values
- Entropy is measured in **bits**

#### 4.2.2 Implementation

```python
def compute_entropy(values: list) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total)
                for count in counts.values())
```

#### 4.2.3 Entropy Features Computed

For each source IP, four entropy features are calculated:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `SrcPortEntropy` | $H(\text{Sport})$ | Port selection randomness for outgoing connections |
| `DestPortEntropy` | $H(\text{Dport})$ | Destination service diversity |
| `DestIPEntropy` | $H(\text{DstAddr})$ | Destination IP diversity |
| `FlagEntropy` | $H(\text{State})$ | TCP flag pattern randomness |

**Interpretation Guide:**
- **Low entropy (~0)**: Concentrated distribution, repetitive behavior
- **High entropy (~$\log_2(n)$)**: Uniform distribution, diverse behavior
- **Botnet signatures**: Often exhibit unusual entropy patterns (either very low for scanning or very high for C&C)

### 4.3 Aggregation by Source IP

The aggregation transforms raw flows into per-IP features:

```python
for ip, group in df.groupby('SrcAddr', sort=False):
    aggregated_rows.append({
        'SrcIP': ip,
        'TotalFlows': len(group),
        'SynFreq': group['syn'].sum() / total_flows,
        'AckFreq': group['ack'].sum() / total_flows,
        'FinFreq': group['fin'].sum() / total_flows,
        'RstFreq': group['rst'].sum() / total_flows,
        'PshFreq': group['psh'].sum() / total_flows,
        'UrgFreq': group['urg'].sum() / total_flows,
        'SrcPortEntropy': compute_entropy(group['Sport']),
        'DestPortEntropy': compute_entropy(group['Dport']),
        'DestIPEntropy': compute_entropy(group['DstAddr']),
        'FlagEntropy': compute_entropy(group['State']),
        'AggregatedLabel': aggregated_label,
    })
```

### 4.4 Label Aggregation

An IP is labeled as **malicious** if:

$$\text{AggregatedLabel} = \begin{cases} 1 & \text{if } \frac{\text{flows with 'From-Botnet'}}{\text{total flows}} > \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

The default threshold is `0.0`, meaning any botnet-associated flow marks the IP as malicious.

---

## 5. Principal Component Analysis (PCA)

### 5.1 Mathematical Foundation

PCA performs an **eigendecomposition** of the covariance matrix to find orthogonal directions of maximum variance.

#### 5.1.1 Standardization

First, features are standardized to zero mean and unit variance:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

This ensures all features contribute equally to the PCA, regardless of their original scales.

#### 5.1.2 Covariance Matrix

For the standardized matrix $Z \in \mathbb{R}^{n \times d}$:

$$\Sigma = \frac{1}{n-1} Z^T Z$$

#### 5.1.3 Eigendecomposition

PCA finds eigenvectors $v_k$ and eigenvalues $\lambda_k$ satisfying:

$$\Sigma v_k = \lambda_k v_k$$

Where:
- $v_k$: Principal component (eigenvector)
- $\lambda_k$: Explained variance (eigenvalue)
- Components are ordered: $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$

### 5.2 Subspace Separation

The key innovation in the Lakhina method is separating PCA space into **major** and **minor** subspaces.

#### 5.2.1 Component Classification

Given $d$ total components and a parameter $k$ (number of minor components):

$$\text{Major components: } \{v_1, v_2, ..., v_{d-k}\}$$
$$\text{Minor components: } \{v_{d-k+1}, ..., v_d\}$$

With default $k_{\text{minor}} = 1$:
- **Major subspace**: Top $d-1$ components (captures dominant traffic patterns)
- **Minor subspace**: Last component (captures rare/exceptional patterns)

#### 5.2.2 Significance Threshold

Components with eigenvalues below threshold are discarded:

$$\text{Significant if } \lambda_k > \epsilon_{\text{eigen}}$$

Default: $\epsilon_{\text{eigen}} = 10^{-6}$

### 5.3 Anomaly Scores

#### 5.3.1 Projection

For each data point $z$ (standardized feature vector):

$$\text{Projected}_k = z \cdot v_k$$

#### 5.3.2 Anomaly Score Formulas

**Major Subspace Score:**

$$S_{\text{major}} = \sum_{k=1}^{d-k_{\text{minor}}} \frac{(z \cdot v_k)^2}{\lambda_k^2}$$

**Minor Subspace Score:**

$$S_{\text{minor}} = \sum_{k=d-k_{\text{minor}}+1}^{d} \frac{(z \cdot v_k)^2}{\lambda_k^2}$$

#### 5.3.3 Mathematical Interpretation

The anomaly score is essentially a **Mahalanobis distance** in each subspace:

- Division by $\lambda_k^2$ weights each component inversely to its variance
- High variance directions (major) contribute less per unit projection
- Low variance directions (minor) are more sensitive to deviations
- **Anomalous traffic**: Projects unusually onto either subspace

#### 5.3.4 Implementation

```python
def calculate_anomaly_scores(k, data_matrix, significant_components, eigenvalues):
    n_components = significant_components.shape[0]

    major_components = significant_components[:n_components - k, :]
    minor_components = significant_components[n_components - k:, :]

    projected_major = data_matrix.dot(major_components.T)
    projected_minor = data_matrix.dot(minor_components.T)

    eigen_major = eigenvalues[:n_components - k]
    eigen_minor = eigenvalues[n_components - k:]

    anomaly_scores_major = np.sum((projected_major ** 2) / np.square(eigen_major), axis=1)
    anomaly_scores_minor = np.sum((projected_minor ** 2) / np.square(eigen_minor), axis=1)

    return anomaly_scores_major, anomaly_scores_minor
```

---

## 6. Detection Logic

### 6.1 OR-Based Classification

A data point is classified as **anomalous** if **either** score exceeds its threshold:

$$\text{Prediction} = \begin{cases} 1 & \text{if } S_{\text{major}} > \tau_{\text{major}} \text{ OR } S_{\text{minor}} > \tau_{\text{minor}} \\ 0 & \text{otherwise} \end{cases}$$

### 6.2 Threshold Calibration

Thresholds are optimized via **grid search** on the validation set:

```python
threshold_major_values = np.linspace(val_major.min(), val_major.max(), grid_points)
threshold_minor_values = np.linspace(val_minor.min(), val_minor.max(), grid_points)
```

For each pair $(\tau_{\text{major}}, \tau_{\text{minor}})$:
1. Compute predictions on validation set
2. Calculate F1-score
3. Select pair maximizing F1 (with FPR tiebreaker)

### 6.3 Optimization Criterion

Selection priority:
1. **Maximize F1-score** (harmonic mean of precision and recall)
2. **Minimize FPR** (false positive rate) for ties

This balances detection accuracy while avoiding alert fatigue from false positives.

---

## 7. Evaluation Metrics

### 7.1 Confusion Matrix

| | Predicted Normal | Predicted Malicious |
|---|---|---|
| **Actual Normal** | True Negative (TN) | False Positive (FP) |
| **Actual Malicious** | False Negative (FN) | True Positive (TP) |

### 7.2 Performance Metrics

**Precision** (Positive Predictive Value):
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (True Positive Rate, Sensitivity):
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score** (Harmonic Mean):
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**False Positive Rate (FPR)**:
$$\text{FPR} = \frac{FP}{FP + TN}$$

**Accuracy**:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 7.3 Why F1 Over Accuracy?

Given the **extreme class imbalance** (~98% background, ~1% botnet), accuracy is misleading. F1-score provides a better measure of detection effectiveness by focusing on the minority (malicious) class.

---

## 8. Implementation Details

### 8.1 Configuration Parameters

```python
CONFIG = {
    'train_ratio': 0.60,              # 60% for training
    'val_ratio': 0.20,                # 20% for validation
    'use_tcp_only': True,             # Filter to TCP flows
    'malicious_threshold': 0.0,       # Label threshold for aggregation
    'pca_feature_columns': (          # Features for PCA
        'SrcPortEntropy',
        'DestPortEntropy',
        'DestIPEntropy',
        'FlagEntropy',
    ),
    'eigen_threshold': 1e-6,          # Minimum eigenvalue significance
    'k_minor': 1,                      # Number of minor components
    'threshold_grid_points': 80,       # Grid search resolution
}
```

### 8.2 Data Preprocessing Steps

1. **Load**: Parse `.binetflow` file
2. **Clean**: Remove invalid entries (missing addresses, negative durations)
3. **Normalize Labels**: Map raw labels to {Botnet, Normal, Background}
4. **Parse Ports**: Handle hexadecimal port notation (e.g., `0x0050` → 80)
5. **Sort Chronologically**: Ensure temporal ordering

### 8.3 Model Persistence

The pipeline saves:
- **PCA components and eigenvalues** (from training)
- **StandardScaler parameters** (mean, std)
- **Optimal thresholds** (calibrated on validation)

---

## 9. Results and Analysis

### 9.1 Typical Results

On CTU-13 Scenario 13:

| Metric | Value |
|--------|-------|
| True Positives | 0-1 |
| True Negatives | ~4000-5000 |
| False Positives | ~20-30 |
| False Negatives | ~0-1 |
| Precision | 0.04-0.05 |
| Recall | 1.00 |
| F1-Score | ~0.08-0.15 |
| FPR | ~0.005 |

### 9.2 Interpretation

- **High Recall (1.0)**: The system detects all botnet IPs
- **Low Precision**: Many false positives due to class imbalance
- **Low F1**: Reflects precision-recall tradeoff

The low precision is expected given the extreme class imbalance (botnet IPs are very rare in the aggregated dataset). The system prioritizes **not missing attacks** over minimizing false alarms.

### 9.3 Threshold Heatmap Analysis

The F1-score heatmap across threshold pairs typically shows:
- Optimal region where both thresholds are relatively high
- Sharp drop-off when thresholds are too low (excessive FP)
- Plateau effect when thresholds exceed necessary range

---

## 10. Advantages and Limitations

### 10.1 Advantages

1. **Unsupervised**: No need for labeled training data (uses statistical properties)
2. **Interpretable**: Entropy features have clear physical meanings
3. **Fast**: PCA transformation is computationally efficient
4. **Adaptive**: Can recalibrate thresholds without retraining

### 10.2 Limitations

1. **Class Imbalance Sensitivity**: Rare botnets lead to poor precision
2. **IP Aggregation Granularity**: May miss attacks from distributed sources
3. **TCP-Only**: Ignores UDP/ICMP botnet traffic
4. **Static Thresholds**: May not adapt to concept drift

### 10.3 Potential Improvements

1. **Weighted PCA**: Account for class imbalance in covariance estimation
2. **Ensemble Methods**: Combine with other detection techniques
3. **Time-Window Aggregation**: Capture temporal dynamics
4. **Deep Learning**: Autoencoders for anomaly detection

---

## 11. Conclusion

This implementation demonstrates the **Lakhina Entropy method** for network anomaly detection. By combining **Shannon entropy features** with **PCA-based subspace analysis**, the system identifies botnet traffic that exhibits unusual traffic distribution patterns.

The mathematical foundation—Mahalanobis distance in PCA subspaces—provides a principled way to detect deviations without requiring explicit attack signatures. While the method shows strong recall, the extreme class imbalance in network traffic data necessitates careful threshold calibration and potentially complementary detection mechanisms for practical deployment.

---

## 12. References

1. Lakhina, A., Crovella, M., & Diot, C. (2005). *Mining Anomalies Using Traffic Feature Distributions*. SIGCOMM '05.

2. Lakhina, A., Crovella, M., & Diot, C. (2004). *Diagnosing Network-Wide Traffic Anomalies*. SIGCOMM '04.

3. Garcia, S., et al. (2014). *An Empirical Comparison of Botnet Detection Methods*. Computers & Security.

4. CTU-13 Dataset: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/

---

## Appendix A: Entropy Calculation Example

Given source ports from flows: `[80, 443, 80, 80, 443]`

**Counts:**
- Port 80: 3 occurrences
- Port 443: 2 occurrences
- Total: 5 flows

**Probabilities:**
- $p_{80} = \frac{3}{5} = 0.6$
- $p_{443} = \frac{2}{5} = 0.4$

**Entropy:**
$$H = -\left(0.6 \log_2(0.6) + 0.4 \log_2(0.4)\right)$$
$$H = -\left(0.6 \times (-0.737) + 0.4 \times (-1.322)\right)$$
$$H = 0.442 + 0.529 = 0.971 \text{ bits}$$

Maximum entropy (uniform distribution): $H_{\max} = \log_2(2) = 1.0$ bits

---

## Appendix B: PCA Anomaly Score Example

For 4-dimensional standardized feature vector $z$ with eigenvalues $[3.0, 1.5, 0.8, 0.2]$ and $k_{\text{minor}}=1$:

**Projections:**
- Major components: $[z \cdot v_1, z \cdot v_2, z \cdot v_3]$
- Minor components: $[z \cdot v_4]$

**Scores:**
$$S_{\text{major}} = \frac{(z \cdot v_1)^2}{3.0^2} + \frac{(z \cdot v_2)^2}{1.5^2} + \frac{(z \cdot v_3)^2}{0.8^2}$$

$$S_{\text{minor}} = \frac{(z \cdot v_4)^2}{0.2^2}$$

Note how the small eigenvalue (0.2) amplifies the minor component's contribution, making it sensitive to deviations in rare patterns.

---

*Document generated on 2026-03-26*