# app.py
# Customer Strategy Simulator — Enhanced with Recommendations, ML Models, Strategies, and Marketing Plan
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.sparse import csr_matrix
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------- Added imports ----------
import os
from scipy import stats

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
def load_aggregated_df(path="customer_data_cleaned (1).csv"):
    df = pd.read_csv(path)
    # Add churn label if not present (example: churn if day_since_last_purchase > 90)
    if 'day_since_last_purchase' in df.columns:
        df['churn'] = (df['day_since_last_purchase'] > 90).astype(int)
    # Add CLV proxy if not present (example: total_spend * (1 + Total_Transactions / 10))
    if 'total_spend' in df.columns and 'Total_Transactions' in df.columns:
        df['clv'] = df['total_spend'] * (1 + df['Total_Transactions'] / 10)
    return df

@st.cache_data
def load_raw_df(path="data.csv"):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df

aggregated_df = load_aggregated_df()
raw_df = load_raw_df()

# Use aggregated for main app
df = aggregated_df

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
# Recommendation System Setup (Collaborative Filtering)
# ---------------------------
@st.cache_data
def build_recommender(raw_df):
    # Aggregate to customer-product matrix (pivot: customers x products, values = quantity)
    customer_product = raw_df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().unstack(fill_value=0)
    customer_product_sparse = csr_matrix(customer_product.values)
    
    # Cosine similarity on customer vectors
    similarity_matrix = cosine_similarity(customer_product_sparse)
    
    return customer_product, similarity_matrix, customer_product.index.tolist()

customer_product, similarity_matrix, customer_ids = build_recommender(raw_df)

def get_recommendations(customer_id, n=5):
    if customer_id not in customer_ids:
        # Fallback: popular items
        popular = raw_df.groupby('StockCode')['Quantity'].sum().nlargest(n).index.tolist()
        rec_details = []
        for stock in popular:
            desc = raw_df[raw_df['StockCode'] == stock]['Description'].iloc[0] if not raw_df[raw_df['StockCode'] == stock].empty else "N/A"
            rec_details.append({'StockCode': stock, 'Description': desc})
        return rec_details
    
    cust_idx = customer_ids.index(customer_id)
    sim_scores = list(enumerate(similarity_matrix[cust_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]  # Top similar customers
    
    rec_products = []
    for idx, score in sim_scores:
        similar_cust_id = customer_ids[idx]
        # Products bought by similar customer but not by this one
        bought_by_sim = set(customer_product.columns[customer_product.loc[similar_cust_id] > 0])
        bought_by_this = set(customer_product.columns[customer_product.loc[customer_id] > 0])
        new_recs = bought_by_sim - bought_by_this
        rec_products.extend(list(new_recs)[:3])  # Limit per similar cust
    
    # Unique, top-N
    rec_products = list(set(rec_products))[:n]
    
    # Fetch descriptions (map back to raw_df)
    rec_details = []
    for stock in rec_products:
        desc = raw_df[raw_df['StockCode'] == stock]['Description'].iloc[0] if not raw_df[raw_df['StockCode'] == stock].empty else "N/A"
        rec_details.append({'StockCode': stock, 'Description': desc})
    
    return rec_details

# --------------------------- 
# Additional ML Models Setup
# ---------------------------
@st.cache_data
def train_ml_models(df, X_scaled):
    # Work on a copy to avoid modifying input df
    local_df = df.copy()
    
    # Churn Prediction (Random Forest)
    if 'churn' in local_df.columns:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, local_df['churn'], test_size=0.2, random_state=42)
        churn_model = RandomForestClassifier(random_state=42)
        churn_model.fit(X_train, y_train)
    else:
        churn_model = None
    
    # CLV Regression (Linear Regression)
    if 'clv' in local_df.columns:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, local_df['clv'], test_size=0.2, random_state=42)
        clv_model = LinearRegression()
        clv_model.fit(X_train, y_train)
    else:
        clv_model = None
    
    # Uplift Modeling (Simplified: Two logistic models for control vs. treatment)
    # Assume 'treatment' as a simulated column (e.g., random for demo; in real, from A/B test)
    local_df['treatment'] = np.random.binomial(1, 0.5, len(local_df))  # Demo
    local_df['response'] = np.random.binomial(1, 0.3 + 0.2 * local_df['treatment'], len(local_df))  # Demo response
    
    mask_control = local_df['treatment'] == 0
    mask_treatment = local_df['treatment'] == 1
    
    X_control = X_scaled[mask_control]
    y_control = local_df[mask_control]['response'].values
    X_treatment = X_scaled[mask_treatment]
    y_treatment = local_df[mask_treatment]['response'].values
    
    control_model = LogisticRegression().fit(X_control, y_control)
    treatment_model = LogisticRegression().fit(X_treatment, y_treatment)
    
    return churn_model, clv_model, control_model, treatment_model

churn_model, clv_model, control_model, treatment_model = train_ml_models(df, X_scaled)

def uplift_score(scaled_vec):
    if control_model is None or treatment_model is None:
        return 0.0
    control_prob = control_model.predict_proba([scaled_vec])[0][1]
    treatment_prob = treatment_model.predict_proba([scaled_vec])[0][1]
    return treatment_prob - control_prob

# --------------------------- 
# Cluster Strategies DataFrame
# ---------------------------
strategies_data = {
    'Cluster': [0, 1, 2],
    'Goal': ['Reactivate (increase txns by 50-100%, reduce recency by 30 days)', 'Increase frequency (boost txns 20-50%, avg txn value 15%)', 'Upsell premium (increase spend 20-30%, unique products +10%)'],
    'Key Tactics': [
        '- Discounted "welcome back" emails on past favorites (20% off first purchase).\n- SMS reminders for abandoned carts.\n- Free shipping on low-value items.',
        '- Loyalty points for repeat buys (e.g., 2x points on next order).\n- Bundles: "Buy 2, get 1 free" on entry-level products.\n- Personalized recs via app.',
        '- Exclusive previews: VIP access to limited-edition items.\n- Premium bundles (e.g., high-margin luxury sets at 10% off).\n- Referral bonuses: "Refer a friend, get £50 credit."'
    ],
    'Expected Impact (Simulate in App)': ['Moves 30-50% to Cluster 1 (via txn/recency levers).', '40% uplift to Cluster 2; simulate with txn/spend sliders.', 'Retain 80% + 10% revenue from upsells; track via monthly spend trend.'],
    'Metrics to Track': ['Reactivation rate, first post-purchase spend.', 'Repeat purchase rate, basket size growth.', 'CLV increase, referral conversions.']
}
strategies_df = pd.DataFrame(strategies_data)

# --------------------------- 
# Marketing Plan Markdown
# ---------------------------
marketing_plan_md = """
### Full-Scale Marketing and Advertisement Strategy (6-Month Phased Plan)

**Overall Framework**:
- **Segmentation**: Use clusters for personalization (80% of campaigns).
- **KPIs**: CAC < £10, ROAS > 3x, 15% cluster migration rate.
- **Tools**: Integrate with Mailchimp/Klaviyo for emails, Google Ads for search, your app for A/B testing levers.

| Phase | Duration | Focus Clusters | Tactics & Channels | Budget Allocation | Expected Outcomes |
|-------|----------|----------------|---------------------|-------------------|-------------------|
| **1: Awareness & Reactivation** | Months 1-2 | Primarily 0 (Dormant), touch 1 | - Email: "Missed You!" series with 20% off past buys + recs (from your system).<br>- SMS: Urgency blasts ("24h flash sale").<br>- Social: Retargeting ads on X/FB for inactive users (lookalikes from Cluster 2).<br>- Content: Blog on "Rediscover Favorites" with product carousels. | 30% (£30K) – Heavy on email/SMS. | 20% reactivation in Cluster 0; 10K new engagements. |
| **2: Engagement & Loyalty Build** | Months 2-4 | Cluster 1 (New/Low), retain 0 movers | - App/Email: Points program (e.g., "Earn £5 per £50 spent") + bundle recs.<br>- Ads: Google search for "affordable gifts" → landing with personalized quizzes (cluster-based).<br>- Influencer: Micro-influencers (10K followers) demo entry bundles on IG/TikTok.<br>- Web: Pop-ups for cart abandonment with txn incentives. | 40% (£40K) – Balanced ads/email. | 30% to Cluster 2 migration; repeat rate +25%. |
| **3: Upsell & Retention** | Months 4-6 | Cluster 2 (High-Value), nurture 1 | - VIP Email: Exclusive drops (premium items) + CLV-based tiers (e.g., "Gold: Free upgrades").<br>- Ads: Dynamic retargeting on Amazon/FB for high-intent (e.g., "Upgrade Your Collection").<br>- Events: Virtual webinars ("Styling Tips") with upsell CTAs.<br>- Loyalty: Referral loops (share recs for credits). | 30% (£30K) – Premium channels (e.g., sponsored X posts). | 15% CLV uplift; 85% retention in Cluster 2. |

**Cross-Cutting Elements**:
- **Personalization**: 100% via recs + cluster (e.g., Cluster 0: Budget items; Cluster 2: Luxury like "VINTAGE BELLS GARLAND").
- **A/B Testing**: Use app sliders to simulate (e.g., test 10% vs. 20% discount on uplift model).
- **Measurement**: Weekly dashboards (Google Analytics + your app metrics). Adjust if Cluster 0 migration <20%.
- **Risks/Mitigation**: Over-discounting erodes margins → Cap at 25%; GDPR compliance for emails.
"""

# --------------------------- 
# Sidebar controls
# ---------------------------
st.sidebar.header("Customer Strategy Controls")

# Core customer selection
customer_id = st.sidebar.selectbox("📊 Select Customer", df['CustomerID'].unique())

# Main strategy levers
st.sidebar.subheader("Strategy Levers")
spend_change = st.sidebar.slider(
    "Marketing Spend", 
    -90, 300, 0, step=5,
    help="Adjust marketing spend to influence purchase behavior"
)
txns_change = st.sidebar.slider(
    "Loyalty Program", 
    -90, 300, 0, step=5,
    help="Modify loyalty incentives to boost repeat purchases"
)
recency_change_days = st.sidebar.slider(
    "Re-engagement Days", 
    -180, 180, 0, step=1,
    help="Negative days = pull customers back sooner"
)

# Cluster-wide toggle
apply_cluster_wide = st.sidebar.checkbox(
    "Apply to Entire Segment", 
    False,
    help="Test strategy on all customers in the same cluster"
)

# Advanced settings
st.sidebar.subheader("Advanced Settings")
st.sidebar.markdown("**    Behavior Multipliers**")
spend_to_avg_txn = st.sidebar.slider(
    "Order Value Impact", 
    0.0, 1.0, 0.3, step=0.05,
    help="How much spend increase affects average order value"
)
txns_to_products = st.sidebar.slider(
    "Cross-Sell Effect", 
    0.0, 1.0, 0.5, step=0.05,
    help="How transactions lead to product diversity"
)
spend_to_monthly = st.sidebar.slider(
    "Monthly Stability", 
    0.0, 1.0, 0.4, step=0.05,
    help="Impact on consistent monthly spending"
)

# Run simulation button with clear call-to-action
simulate_button = st.sidebar.button("Run Strategy Simulation", type="primary")

# --------------------------- 
# Behavioral cascade function
# ---------------------------
def apply_behavioral_cascade(row, spend_pct, txns_pct, recency_days,
                             spend_to_avg=0.3, txns_to_prod=0.5, spend_to_monthly=0.4):
    modified = row.copy().astype(float)
    modified['total_spend'] = modified['total_spend'] * (1 + spend_pct / 100)
    modified['Total_Transactions'] = modified['Total_Transactions'] * (1 + txns_pct / 100)
    if 'day_since_last_purchase' in modified.index:
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

def effectiveness_score(dist_before, dist_after, orig_cluster, new_cluster):
    """
    Enhanced effectiveness score that prioritizes:
    1. Cluster upgrades (major positive impact)
    2. Distance improvement within same cluster (minor positive impact)
    3. Cluster downgrades (negative impact)
    
    Returns score between -100 and 100:
    - Positive: Better cluster or closer to better centroid
    - Negative: Worse cluster or further from ideal centroid
    """
    # Cluster movement impact (primary factor)
    cluster_diff = new_cluster - orig_cluster
    
    # Base score from cluster movement
    if cluster_diff > 0:  # Upgrade
        cluster_score = 50 * cluster_diff  # +50 per cluster upgrade
    elif cluster_diff < 0:  # Downgrade
        cluster_score = 60 * cluster_diff  # -60 per cluster downgrade
    else:  # Same cluster
        cluster_score = 0
    
    # Distance component (secondary factor)
    # For upgrades: distance to new centroid matters more than distance improvement
    if cluster_diff > 0:
        # When upgrading, we care more about being close to new cluster
        dist_score = 20 * (1 - dist_after / (dist_before + 1e-6))
    else:
        # When same/downgrade, we care about relative improvement
        dist_score = 30 * ((dist_before - dist_after) / (dist_before + 1e-6))
    
    # Combine scores
    final_score = cluster_score + dist_score
    
    # Ensure upgrades are always positive
    if cluster_diff > 0:
        final_score = max(20, final_score)  # Minimum 20% score for any upgrade
        
    return max(-100, min(100, final_score))

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
# Personalized Recommendations
# ---------------------------
st.subheader("Personalized Recommendations")
recs = get_recommendations(customer_id, n=5)
if recs:
    rec_df = pd.DataFrame(recs)
    st.dataframe(rec_df, use_container_width=True)
    st.write(f"Based on similar customers to ID {customer_id}. Encourage upsell with these!")
else:
    st.info("No recommendations available—customer has unique profile.")

# --------------------------- 
# Additional ML Predictions (Tabs)
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Churn Risk", "CLV Forecast", "Uplift Score"])
cust_idx = df[df['CustomerID'] == customer_id].index[0]
orig_scaled_vec = df_scaled.loc[cust_idx, features].values

with tab1:
    if churn_model:
        churn_prob = churn_model.predict_proba([orig_scaled_vec])[0][1]
        st.metric("Churn Probability", f"{churn_prob:.2%}")
        if churn_prob > 0.5:
            st.warning("High churn risk—prioritize retention tactics.")
    else:
        st.info("Churn model not available (add 'churn' column).")

with tab2:
    if clv_model:
        clv_pred = clv_model.predict([orig_scaled_vec])[0]
        st.metric("Projected CLV", f"₹{clv_pred:,.2f}")
    else:
        st.info("CLV model not available (add 'clv' column).")

with tab3:
    uplift = uplift_score(orig_scaled_vec)
    st.metric("Uplift Score (Treatment Effect)", f"{uplift:.2%}")
    if uplift > 0.1:
        st.success("High potential uplift—target with campaigns.")

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
    if 'day_since_last_purchase' in orig_row.index:
        cust_recency = orig_row['day_since_last_purchase']
    cust_monthly_mean = orig_row['Monthly_Spending_Mean']

    # Dataset-level averages for normalization
    avg_spend = df['total_spend'].mean()
    avg_txns = df['Total_Transactions'].mean()
    avg_products = df['Total_Products_Purchased'].mean()
    if 'day_since_last_purchase' in df.columns:
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
    if 'day_since_last_purchase' in orig_row.index:
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
        
        # Calculate effectiveness
        eff = effectiveness_score(
            orig_nearest_dist,
            new_nearest_dist,
            int(orig_row['cluster']),
            new_pred
        )

        # Display metrics
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Original Cluster", f"{int(orig_row['cluster'])}")
        col_b.metric("New Cluster (Predicted)", f"{int(new_pred)}")
        
        # Add delta indicator for cluster movement
        delta = new_pred - int(orig_row['cluster'])
        col_c.metric("Strategy Effectiveness", 
                     f"{eff:.1f}%",
                     delta=f"{'+' if delta > 0 else ''}{delta} clusters" if delta != 0 else "Same cluster")

        # Enhanced movement interpretation
        st.markdown("### Strategy Impact Analysis")
        
        if new_pred != int(orig_row['cluster']):
            if new_pred > int(orig_row['cluster']):
                st.success(f"🎯 Cluster Upgrade: {cluster_name[int(orig_row['cluster'])]} → {cluster_name[new_pred]}")
                
                value_increase = (new_pred - int(orig_row['cluster'])) * 25
                st.markdown(f"""
                ### Positive Movement Analysis
                - **Cluster Improvement**: +{new_pred - int(orig_row['cluster'])} levels
                - **Expected Value Increase**: ~{value_increase}%
                - **Distance Quality**: {' Good' if new_nearest_dist < orig_nearest_dist else ' Needs Optimization'} fit to new cluster
                
                **Next Steps:**
                1. Apply retention strategies for {cluster_name[new_pred]}
                2. Monitor for stability in new segment
                3. Consider additional uplift opportunities
                """)
                
            else:
                st.warning(f"⚠️ Cluster Downgrade: {cluster_name[int(orig_row['cluster'])]} → {cluster_name[new_pred]}")
                st.markdown(f"""
                ### Movement Analysis (Caution)
                - **Cluster Change**: {new_pred - int(orig_row['cluster'])} levels
                - **Risk Level**: High - Immediate attention needed
                - **Recovery Plan**: Review strategy recommendations below
                """)
        else:
            next_cluster = min(2, int(orig_row['cluster']) + 1)
            progress = ((orig_distances[next_cluster] - new_dists[next_cluster]) / orig_distances[next_cluster] * 100)
            
            if progress > 0:
                st.info(f"📈 Progress: {progress:.1f}% closer to next cluster upgrade")
            else:
                st.info("⚖️ Stable within current cluster")
    
        # Updated effectiveness interpretation
        st.markdown("### Strategy Effectiveness Analysis")
        if eff > 50:
            st.success(f"""
            🌟 **Outstanding Results** (Score: {eff:.1f}%)
            - Strong cluster upgrade achieved
            - Good positioning within new segment
            - Ready for advanced strategies
            """)
        elif eff > 25:
            st.success(f"""
            ✨ **Positive Impact** (Score: {eff:.1f}%)
            - Clear improvement demonstrated
            - Good strategic direction
            - Continue current approach
            """)
        elif eff > 0:
            st.info(f"""
            📊 **Moderate Progress** (Score: {eff:.1f}%)
            - Some positive movement detected
            - Consider strengthening interventions
            - Review strategy mix
            """)
        else:
            st.warning(f"""
            ⚠️ **Strategy Alert** (Score: {eff:.1f}%)
            - Movement not optimal
            - Review customer profile
            - Consider alternative approaches
            """)

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
        modified_scaled_vec = scale_row_for_features(modified_df.iloc[0])
        mod_point = pca.transform([modified_scaled_vec])[0]
        fig.add_trace(go.Scatter(x=[orig_point[0]], y=[orig_point[1]],
                                 mode='markers', marker=dict(size=12, color='green'), name='Original'))
        fig.add_trace(go.Scatter(x=[mod_point[0]], y=[mod_point[1]],
                                 mode='markers', marker=dict(size=14, color='red'), name='Modified'))
        fig.add_trace(go.Scatter(x=[orig_point[0], mod_point[0]], y=[orig_point[1], mod_point[1]],
                                 mode='lines', line=dict(color='gray', width=2), showlegend=False))

    fig.update_layout(height=600, legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # Post-Simulation Recommendations
    st.subheader("Post-Simulation Recommendations")
    # Simulate updated row for recs (use modified_df.iloc[0])
    updated_recs = get_recommendations(customer_id, n=5)  # Re-run (in real, adjust matrix)
    if updated_recs:
        updated_rec_df = pd.DataFrame(updated_recs)
        st.dataframe(updated_rec_df, use_container_width=True)
    else:
        st.info("No updated recommendations.")

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

# --------------------------- 
# Tailored Strategies Expander
# ---------------------------
with st.expander("View Tailored Strategies"):
    cluster = int(df[df['CustomerID'] == customer_id]['cluster'].iloc[0])
    st.subheader(f"Strategies for {cluster_name.get(cluster, f'Cluster {cluster}')}")
    filtered_strategies = strategies_df[strategies_df['Cluster'] == cluster]
    st.table(filtered_strategies)

# --------------------------- 
# Marketing Plan Expander
# ---------------------------
with st.expander("Full Marketing & Advertisement Strategy"):
    st.markdown(marketing_plan_md)

# ---------- New helper: A/B testing simulator ----------
def ab_test_simulation(base_row, variant_a, variant_b, n_boot=200):
    """
    base_row: pd.Series original customer row
    variant_a/b: dict with keys spend_pct, txns_pct, recency_days
    Returns summary dict with mean uplift estimates and p-value (bootstrap).
    """
    # Apply deterministic change to get point estimates
    a_row = apply_behavioral_cascade(base_row, variant_a['spend_pct'], variant_a['txns_pct'], variant_a['recency_days'],
                                     spend_to_avg=variant_a.get('spend_to_avg', 0.3),
                                     txns_to_prod=variant_a.get('txns_to_products', 0.5),
                                     spend_to_monthly=variant_a.get('spend_to_monthly', 0.4))
    b_row = apply_behavioral_cascade(base_row, variant_b['spend_pct'], variant_b['txns_pct'], variant_b['recency_days'],
                                     spend_to_avg=variant_b.get('spend_to_avg', 0.3),
                                     txns_to_prod=variant_b.get('spend_to_products', 0.5),
                                     spend_to_monthly=variant_b.get('spend_to_monthly', 0.4))
    # Scale for model input
    a_scaled = scale_row_for_features(a_row)
    b_scaled = scale_row_for_features(b_row)
    base_scaled = scale_row_for_features(base_row)
    # Use uplift_score (treatment vs control model) if available, else fallback to churn probability delta
    def estimate_effect(scaled_vec):
        try:
            return uplift_score(scaled_vec)
        except Exception:
            # fallback: difference in churn risk (negative uplift = lower churn -> better)
            if churn_model is not None:
                return - (churn_model.predict_proba([scaled_vec])[0][1])
            return 0.0

    est_base = estimate_effect(base_scaled)
    est_a = estimate_effect(a_scaled)
    est_b = estimate_effect(b_scaled)

    # Bootstrap synth noise around scaled vector to compute distribution
    rng = np.random.default_rng(42)
    a_samps = []
    b_samps = []
    for _ in range(n_boot):
        jitter = rng.normal(scale=0.02, size=len(a_scaled))  # small noise
        a_val = estimate_effect(a_scaled + jitter)
        jitter = rng.normal(scale=0.02, size=len(b_scaled))
        b_val = estimate_effect(b_scaled + jitter)
        a_samps.append(a_val)
        b_samps.append(b_val)

    # t-test
    try:
        tstat, pval = stats.ttest_ind(a_samps, b_samps, equal_var=False)
    except Exception:
        pval = 1.0

    summary = {
        'base': est_base,
        'variant_a': est_a,
        'variant_b': est_b,
        'a_samples_mean': np.mean(a_samps),
        'b_samples_mean': np.mean(b_samps),
        'p_value': float(pval)
    }
    return summary

# ---------- New helper: Cohort builder ----------
def build_cohort_table(raw_df, aggregated_df, date_col='InvoiceDate'):
    """
    Constructs a cohort-month vs current cluster table (percentage distribution).
    Expects raw_df with InvoiceDate and CustomerID; aggregated_df with CustomerID and cluster.
    Returns pivot_df (cohort_month x cluster) as percentage.
    """
    if date_col not in raw_df.columns or 'CustomerID' not in raw_df.columns:
        return None
    try:
        tmp = raw_df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
        first_purchase = tmp.groupby('CustomerID')[date_col].min().dropna().to_frame('first_date')
        first_purchase['cohort_month'] = first_purchase['first_date'].dt.to_period('M').astype(str)
        merged = first_purchase.merge(aggregated_df[['CustomerID', 'cluster']], left_index=True, right_on='CustomerID', how='left')
        pivot = pd.crosstab(merged['cohort_month'], merged['cluster'], normalize='index') * 100
        pivot = pivot.sort_index()
        return pivot
    except Exception:
        return None

# ---------- New helper: GenAI-style summary (local fallback) ----------
def generate_ai_insights(sim_summary, target_customer_id, use_external=False):
    """
    Returns a concise narrative insight. If use_external True and GEMINI_API_KEY present,
    you may implement an external call (not provided here). This function provides local auto-generated text.
    """
    if sim_summary is None:
        return "No simulation run yet to summarize."
    base = sim_summary.get('base')
    a = sim_summary.get('variant_a')
    b = sim_summary.get('variant_b')
    p = sim_summary.get('p_value', None)
    lines = []
    lines.append(f"Customer {target_customer_id} — Quick AI Summary:")
    lines.append(f"- Baseline estimated uplift: {base:.2%}")
    lines.append(f"- Variant A estimated uplift: {a:.2%}")
    lines.append(f"- Variant B estimated uplift: {b:.2%}")
    if p is not None:
        lines.append(f"- Comparative p-value (A vs B): {p:.4f}")
        if p < 0.05:
            lines.append("- Result: Statistically significant difference — prefer the better-performing variant.")
        else:
            lines.append("- Result: No statistically significant difference — consider larger sample or stronger levers.")
    # Recommendation heuristic
    better = 'A' if sim_summary.get('variant_a', 0) > sim_summary.get('variant_b', 0) else 'B'
    lines.append(f"- Recommendation: Run a controlled A/B test at scale on {better} (pilot n=1k customers) and monitor CLV + churn.")
    return "\n".join(lines)

# ---------- Inserted UI: A/B Test Simulator expander ----------
with st.expander("A/B Test Simulator"):
    st.write("Compare two variants (e.g., discount vs. loyalty) using your uplift model. Uses the selected customer as a template.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Variant A - Quick inputs")
        a_spend = st.number_input("A: Spend % change", value=10, step=1)
        a_txns = st.number_input("A: Txns % change", value=10, step=1)
        a_recency = st.number_input("A: Recency days change", value=-14, step=1)
    with col2:
        st.write("Variant B - Quick inputs")
        b_spend = st.number_input("B: Spend % change", value=0, step=1)
        b_txns = st.number_input("B: Txns % change", value=20, step=1)
        b_recency = st.number_input("B: Recency days change", value=-7, step=1)
    ab_boot = st.slider("Bootstrap samples (quality vs speed)", 50, 1000, 200, step=50)

    run_ab = st.button("Run A/B Simulation")
    last_ab_summary = None
    if run_ab:
        # get customer row
        if customer_id not in df['CustomerID'].values:
            st.error("Selected customer not found for A/B simulation.")
        else:
            idx = df[df['CustomerID'] == customer_id].index[0]
            base_row = df.loc[idx]
            variant_a = {'spend_pct': a_spend, 'txns_pct': a_txns, 'recency_days': a_recency}
            variant_b = {'spend_pct': b_spend, 'txns_pct': b_txns, 'recency_days': b_recency}
            with st.spinner("Simulating variants..."):
                summary = ab_test_simulation(base_row, variant_a, variant_b, n_boot=ab_boot)
                last_ab_summary = summary
            # Visualize
            bars = pd.DataFrame({
                'Scenario': ['Baseline', 'Variant A', 'Variant B'],
                'Estimated Uplift': [summary['base'], summary['variant_a'], summary['variant_b']]
            })
            fig_ab = px.bar(bars, x='Scenario', y='Estimated Uplift', color='Scenario', text='Estimated Uplift',
                            labels={'Estimated Uplift': 'Estimated Uplift (treatment effect)'},
                            title='A/B Estimated Uplift Comparison')
            fig_ab.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            st.plotly_chart(fig_ab, use_container_width=True)
            st.write(f"P-value (A vs B): {summary['p_value']:.4f}")
            if summary['p_value'] < 0.05:
                st.success("Significant difference detected between variants.")
            else:
                st.info("No significant difference detected. Consider larger n or stronger levers.")
            # store for AI insights section
            st.session_state['last_ab_summary'] = summary

# ---------- Inserted UI: Dynamic Cohort Analysis ----------
with st.expander("Dynamic Cohort Analysis"):
    st.write("Cohort month (first purchase) vs current cluster distribution. Useful to spot retention / migration patterns.")
    cohort_tbl = build_cohort_table(raw_df, df, date_col='InvoiceDate' if 'InvoiceDate' in raw_df.columns else 'InvoiceDate')
    if cohort_tbl is None or cohort_tbl.empty:
        st.info("Cohort analysis not available: raw transaction date column 'InvoiceDate' or CustomerID may be missing or unparsable in raw_df.")
    else:
        st.write("Heatmap: % customers in each cluster per cohort month (rows = cohort month).")
        fig_cohort = px.imshow(cohort_tbl.fillna(0).T, labels=dict(x="Cohort Month", y="Cluster", color="% of cohort"), aspect="auto",
                               title="Cohort -> Current Cluster Distribution (%)")
        st.plotly_chart(fig_cohort, use_container_width=True)
        st.dataframe(cohort_tbl.round(2))

# ---------- Inserted UI: GenAI-Powered Insights (local fallback) ----------
with st.expander("GenAI-Powered Insights (Auto Summary)"):
    st.write("Auto-generated narratives for the last A/B simulation or the last run simulation. External GenAI can be enabled via GEMINI_API_KEY env var (do NOT paste keys into the app).")
    sim_summary = st.session_state.get('last_ab_summary', None)
    if sim_summary is None:
        st.info("Run the A/B Simulator to generate insights, or run the main simulation and use the interpretation pane.")
    else:
        use_external = False
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            st.write("External GenAI key detected in environment — external summarization can be enabled (not automatic in this demo).")
            # For safety we do not auto-call external services here; keep local summary for now.
            use_external = False
        insight_text = generate_ai_insights(sim_summary, customer_id, use_external=use_external)
        st.text_area("AI Summary", value=insight_text, height=220)