# app.py
# Customer Strategy Simulator — Improved UI + Cluster Names + Animation
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Customer Strategy Simulator", layout="wide")
st.title("Customer Strategy Simulator")
st.markdown(
    "Simulate strategy levers (spend, transactions, recency). Visualize movement in PCA space and measure Strategy Effectiveness."
)

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_df(path="customer_data_cleaned (1).csv"):
    df = pd.read_csv(path)
    return df

df = load_df()
st.sidebar.header("Dataset Info")
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------------------------
# Configure features & sanity checks
# ---------------------------
features = [
    'Total_Transactions', 'Total_Products_Purchased', 'total_spend',
    'Average_Transaction_Value', 'Unique_Products_Purchased',
    'Average_Days_Between_Purchases', 'Cancellation_Frequency',
    'Cancellation_Rate', 'Monthly_Spending_Mean',
    'Monthly_Spending_Std', 'Spending_Trend'
]

missing = [c for c in features if c not in df.columns]
if missing:
    st.error("Missing expected feature columns: " + ", ".join(missing))
    st.stop()

# ---------------------------
# Cluster name mapping (for clarity)
# ---------------------------
cluster_name = {
    0: "Dormant / Inactive Customers",
    1: "New / Low-Value Customers",
    2: "High-Value / Loyal Customers"
}

# colors for clusters
cluster_colors = {
    0: "lightgray",
    1: "royalblue",
    2: "goldenrod"
}

# ---------------------------
# Standardize and compute centroids
# ---------------------------
scaler = StandardScaler()
X = df[features].astype(float)
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)
df_scaled['cluster'] = df['cluster'].values

cluster_centroids = df_scaled.groupby('cluster').mean()

# ---------------------------
# PCA (2D) for visualization
# ---------------------------
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'], index=df.index)
df_pca['cluster'] = df['cluster'].values
df_pca['cluster_name'] = df_pca['cluster'].map(cluster_name)
centroids_pca = df_pca.groupby('cluster')[['PC1', 'PC2']].mean().reset_index()
centroids_pca['cluster_name'] = centroids_pca['cluster'].map(cluster_name)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Simulation Controls")
customer_id = st.sidebar.selectbox("Select CustomerID", df['CustomerID'].unique())
spend_change = st.sidebar.slider("Change total_spend (%)", -90, 300, 0, step=5)
txns_change = st.sidebar.slider("Change Total_Transactions (%)", -90, 300, 0, step=5)
recency_change_days = st.sidebar.slider(
    "Change day_since_last_purchase (days, negative = more recent)", -180, 180, 0, step=1
)
apply_cluster_wide = st.sidebar.checkbox("Apply to entire selected customer's cluster (cluster-wide)", False)

# Optional cascade intensity sliders (not required, default sensible)
st.sidebar.markdown("**Cascade intensity (how changes ripple to related metrics)**")
spend_to_avg_txn = st.sidebar.slider("Spend → Average Transaction Value multiplier", 0.0, 1.0, 0.3, step=0.05)
txns_to_products = st.sidebar.slider("Txns → Total Products multiplier", 0.0, 1.0, 0.5, step=0.05)
spend_to_monthly = st.sidebar.slider("Spend → Monthly Mean multiplier", 0.0, 1.0, 0.4, step=0.05)

simulate_button = st.sidebar.button("Run Simulation")

# ---------------------------
# Behavioral cascade function
# ---------------------------
def apply_behavioral_cascade(row, spend_pct, txns_pct, recency_days,
                             spend_to_avg=0.3, txns_to_prod=0.5, spend_to_monthly=0.4):
    modified = row.copy().astype(float)
    modified['total_spend'] = modified['total_spend'] * (1 + spend_pct / 100)
    modified['Total_Transactions'] = modified['Total_Transactions'] * (1 + txns_pct / 100)
    modified['day_since_last_purchase'] = max(0, modified['day_since_last_purchase'] + recency_days)
    # propagate to average txn value
    if modified['Total_Transactions'] > 0:
        modified['Average_Transaction_Value'] = modified['total_spend'] / max(1, modified['Total_Transactions'])
    else:
        modified['Average_Transaction_Value'] *= (1 + spend_to_avg * spend_pct / 100)
    # propagate to total products
    modified['Total_Products_Purchased'] = modified['Total_Products_Purchased'] * (1 + txns_to_prod * txns_pct / 100)
    # monthly mean
    modified['Monthly_Spending_Mean'] = modified['Monthly_Spending_Mean'] * (1 + spend_to_monthly * spend_pct / 100)
    # cancellation rate soft decline when spend/txns increase
    if spend_pct > 0 or txns_pct > 0:
        modified['Cancellation_Rate'] = max(0, modified['Cancellation_Rate'] * (1 - 0.1 * (spend_pct/100 + txns_pct/100)))
    return modified

# ---------------------------
# Utility functions
# ---------------------------
def predict_cluster_and_distances_from_scaled_vector(scaled_vec):
    distances = cluster_centroids.apply(lambda c: np.linalg.norm(scaled_vec - c.values), axis=1)
    nearest = int(distances.idxmin())
    return nearest, distances.to_dict()

def effectiveness_score(dist_before, dist_after):
    if dist_before <= 0:
        return 0.0
    return float((dist_before - dist_after) / dist_before * 100)

def scale_row_for_features(row):
    vals = [row[f] if f in row.index else df[f].mean() for f in features]
    return scaler.transform([vals])[0]

# ---------------------------
# Show high-level cluster summary cards
# ---------------------------
st.subheader("Cluster Overview (at a glance)")
col_counts = st.columns(3)
for i, c in enumerate(sorted(df['cluster'].unique())):
    name = cluster_name.get(c, f"Cluster {c}")
    color = cluster_colors.get(c, "gray")
    cnt = int((df['cluster'] == c).sum())
    avg_spend = df.loc[df['cluster'] == c, 'total_spend'].mean()
    contrib = (df.loc[df['cluster'] == c, 'total_spend'].sum() / df['total_spend'].sum()) * 100
    with col_counts[i]:
        st.markdown(f"**{c}: {name}**")
        st.metric("Customers", f"{cnt:,}")
        st.write(f"Avg spend: ${avg_spend:,.2f}" if not np.isnan(avg_spend) else "Avg spend: N/A")
        st.write(f"Revenue contribution: {contrib:.1f}%")

st.markdown("---")

# ---------------------------
# Run simulation when requested
# ---------------------------
if simulate_button:
    cust_mask = df['CustomerID'] == customer_id
    if cust_mask.sum() == 0:
        st.error("Selected Customer ID not found.")
        st.stop()

    cust_idx = df[cust_mask].index[0]
    orig_row = df.loc[cust_idx]

    # These are calculated once per selected customer
    cust_spend = orig_row['total_spend']
    cust_txns = orig_row['Total_Transactions']
    cust_products = orig_row['Total_Products_Purchased']
    cust_recency = orig_row['day_since_last_purchase']
    cust_monthly_mean = orig_row['Monthly_Spending_Mean']

    # Dataset-level averages for normalization
    avg_spend = df['total_spend'].mean()
    avg_txns = df['Total_Transactions'].mean()
    avg_products = df['Total_Products_Purchased'].mean()
    avg_recency = df['day_since_last_purchase'].mean()
    avg_monthly_mean = df['Monthly_Spending_Mean'].mean()

    # Derived behavioral indicators (non-synthetic)
    spend_to_avg = cust_spend / avg_spend
    txns_to_products = cust_txns / (cust_products if cust_products > 0 else 1)
    spend_to_monthly = cust_spend / (cust_monthly_mean if cust_monthly_mean > 0 else avg_monthly_mean)

    # Replace synthetic dependencies with real derived ratios
    spend_to_avg = orig_row['total_spend'] / avg_spend

    txns_to_products = (
        orig_row['Total_Transactions'] / orig_row['Total_Products_Purchased']
        if orig_row['Total_Products_Purchased'] > 0 else 0
    )

    # Recency-adjusted monthly spending intensity
    spend_to_monthly = (
        orig_row['total_spend'] / ((orig_row['day_since_last_purchase'] / 30) + 1)
    )


    # Build modified rows
    if apply_cluster_wide:
        target_cluster = int(orig_row['cluster'])
        affected_idx = df[df['cluster'] == target_cluster].index
        modified_rows = [
            apply_behavioral_cascade(df.loc[i], spend_change, txns_change, recency_change_days,
                                     spend_to_avg, txns_to_products, spend_to_monthly)
            for i in affected_idx
        ]
        modified_df = pd.DataFrame(modified_rows, index=affected_idx)
    else:
        modified_series = apply_behavioral_cascade(orig_row, spend_change, txns_change, recency_change_days,
                                                   spend_to_avg, txns_to_products, spend_to_monthly)
        modified_df = pd.DataFrame([modified_series], index=[cust_idx])

    # Original scaled vector and distances
    orig_scaled_vec = df_scaled.loc[cust_idx, features].values
    orig_pred_cluster, orig_distances = predict_cluster_and_distances_from_scaled_vector(orig_scaled_vec)
    orig_nearest_dist = min(orig_distances.values())

    # For each modified row, scale and predict
    modified_preds = []
    modified_dist_dicts = []
    for idx, row in modified_df.iterrows():
        scaled_vec = scale_row_for_features(row)
        pred_cluster, dists = predict_cluster_and_distances_from_scaled_vector(scaled_vec)
        modified_preds.append((idx, pred_cluster))
        modified_dist_dicts.append(dists)

    # Display results
    st.header("Simulation Result")
    if not apply_cluster_wide:
        new_pred = modified_preds[0][1]
        new_dists = modified_dist_dicts[0]
        new_nearest_dist = min(new_dists.values())
        eff = effectiveness_score(orig_nearest_dist, new_nearest_dist)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Original Cluster", f"{int(orig_row['cluster'])}")
        col_b.metric("New Cluster (Predicted)", f"{int(new_pred)}")
        col_c.metric("Strategy Effectiveness", f"{eff:.2f}%")
        if new_pred != int(orig_row['cluster']):
            st.success("Customer moved to a different segment after the simulated changes.")
        else:
            st.info("Customer remained in the same segment after the simulated changes.")

        # Distance tables
        st.markdown("**Distances to Centroids (Before)**")
        st.dataframe(pd.Series(orig_distances).rename("distance_before").sort_values().to_frame())
        st.markdown("**Distances to Centroids (After)**")
        st.dataframe(pd.Series(new_dists).rename("distance_after").sort_values().to_frame())

    else:
        orig_clusters = df_scaled.loc[modified_df.index, 'cluster'].values
        new_clusters = np.array([p for _, p in modified_preds])
        mobility = pd.crosstab(pd.Series(orig_clusters, name='orig'), pd.Series(new_clusters, name='new'),
                               normalize='index') * 100
        st.subheader("Cluster Mobility (%) — Rows = Original Cluster")
        st.dataframe(mobility.round(2))
        moves = (orig_clusters != new_clusters).sum()
        st.write(f"{moves} / {len(orig_clusters)} customers ({(moves / len(orig_clusters) * 100):.2f}%) changed cluster.")

    # ---------------------------
    # PCA movement animation (interactive)
    # ---------------------------
    st.header("PCA Movement (Interactive)")

    fig = px.scatter(
        df_pca.reset_index(),
        x='PC1', y='PC2',
        color='cluster_name',
        color_discrete_map={cluster_name[k]: cluster_colors[k] for k in cluster_colors},
        hover_data={'index': True, 'cluster': True},
        title="Customers in PCA Space (PC1 vs PC2)"
    )

    # Add centroids as large markers
    for _, r in centroids_pca.iterrows():
        fig.add_trace(go.Scatter(
            x=[r['PC1']], y=[r['PC2']],
            mode='markers+text',
            marker=dict(symbol='x', size=18, color='black'),
            text=[r['cluster_name']],
            textposition='top center',
            showlegend=False
        ))

    # Original and modified points
    if apply_cluster_wide:
        sample_idx = modified_df.index[:50]
        orig_points = pca.transform(df_scaled.loc[sample_idx, features])
        mod_points = np.array([scale_row_for_features(modified_df.loc[i]) for i in sample_idx])
        mod_points = pca.transform(mod_points)
        for o, m in zip(orig_points, mod_points):
            fig.add_trace(go.Scatter(x=[o[0], m[0]], y=[o[1], m[1]],
                                     mode='lines+markers', marker=dict(size=6, color='red'),
                                     line=dict(color='gray', width=1), hoverinfo='none', showlegend=False))
    else:
        orig_point = pca.transform([orig_scaled_vec])[0]
        modified_scaled_vec = modified_preds and scale_row_for_features(modified_df.iloc[0])
        mod_point = pca.transform([modified_scaled_vec])[0]
        fig.add_trace(go.Scatter(x=[orig_point[0]], y=[orig_point[1]],
                                 mode='markers', marker=dict(size=12, color='black'), name='Original'))
        fig.add_trace(go.Scatter(x=[mod_point[0]], y=[mod_point[1]],
                                 mode='markers', marker=dict(size=14, color='red'), name='Modified'))
        fig.add_trace(go.Scatter(x=[orig_point[0], mod_point[0]], y=[orig_point[1], mod_point[1]],
                                 mode='lines', line=dict(color='gray', width=2), showlegend=False))

    fig.update_layout(height=600, legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation section
    st.markdown("---")
    st.subheader("Interpretation & Recommendation")
    if not apply_cluster_wide:
        if eff > 15:
            st.success("Strategy looks effective — the customer moved substantially toward a different behavioral profile.")
        elif eff > 2:
            st.info("Small but measurable movement toward target centroid. Consider increasing budget or combining levers.")
        else:
            st.warning("Minimal movement. This customer's profile is rigid — consider alternative levers (recency or product offers).")
        st.markdown(
            f"- **Effectiveness Score:** {eff:.2f}% (Reduction in distance to nearest centroid)\n"
            f"- **What Worked:** Adjusting spend and/or transactions shifted this profile in feature space.\n"
            f"- **Next Step:** Combine multiple levers (spend + recency + transactions) for maximum impact."
        )
    else:
        st.markdown(
            "- Cluster-wide simulation reveals how a campaign targeted at this segment redistributes customers across clusters.\n"
            "- Ideal for high-level strategy (e.g., loyalty campaigns, targeted incentives)."
        )


