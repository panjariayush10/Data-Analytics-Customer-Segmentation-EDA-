# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
sns.set_style("whitegrid")
st.title("🎯 Customer Segmentation — Full Dashboard")

# ---- Load dataset (local file) ----
DATA_PATH = "customers.dataset.xlsx"  # put the file in the same folder as app.py
try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"Could not read '{DATA_PATH}'. Make sure the file exists in the same folder as app.py.\n\nError: {e}")
    st.stop()

# Basic cleaning
df = df.copy()
df = df.drop_duplicates().reset_index(drop=True)

# Infer numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("No numeric columns detected in dataset. Clustering requires numeric features.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
k = st.sidebar.slider("Number of clusters (k)", 2, 12, 3)
scale_features = st.sidebar.checkbox("Standardize numeric features (recommended)", value=True)
use_cols = st.sidebar.multiselect("Select numeric features to use for clustering / plots", numeric_cols, default=numeric_cols)
sample_for_pairplot = st.sidebar.slider("Pairplot sample size (0 = use all)", 0, 2000, 500)
run_elbow = st.sidebar.checkbox("Show Elbow & Silhouette analysis", value=True)
show_3d = st.sidebar.checkbox("Enable 3D PCA scatter (if 3D capable)", value=False)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# Basic info & preview
tab_overview, tab_preproc, tab_clusters, tab_plots, tab_stats = st.tabs([
    "Overview", "Preprocessing", "Clustering & Eval", "All Charts", "Export & Summary"
])

with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100))
    st.markdown(f"**Rows:** {df.shape[0]} — **Columns:** {df.shape[1]}")
    with st.expander("Show column types & non-null counts"):
        st.dataframe(df.dtypes.rename("dtype").to_frame().join(df.notnull().sum().rename("non-null")))

    st.subheader("Statistical summary (numeric)")
    st.dataframe(df[numeric_cols].describe().T)

with tab_preproc:
    st.subheader("Preprocessing")
    st.write("Selected numeric features:", use_cols)

    X = df[use_cols].copy()
    # handle missing values (simple approach: drop rows with missing in selected features)
    missing_before = X.isna().sum().sum()
    if missing_before > 0:
        st.warning(f"Found {missing_before} missing values in selected features — rows with missing values will be dropped for clustering.")
        X = X.dropna()
    else:
        st.success("No missing values in selected features.")

    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    st.write(f"Data used for clustering: {X.shape[0]} rows × {X.shape[1]} features")

with tab_clusters:
    st.subheader("Run KMeans & Evaluation")
    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_clustered = df.loc[X.index].copy()
    df_clustered["Cluster"] = labels

    st.metric("Cluster counts (largest → smallest)", value="")
    counts = df_clustered["Cluster"].value_counts().sort_values(ascending=False)
    st.dataframe(counts.rename("count").to_frame())

    # Silhouette
    sil_score = None
    try:
        if len(set(labels)) > 1 and len(X_scaled) > k:
            sil_score = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette Score: **{sil_score:.4f}**  — higher is better (max 1.0)")
        else:
            st.info("Silhouette score not computable (need >1 cluster and samples > n_clusters).")
    except Exception as e:
        st.warning(f"Silhouette score failed: {e}")

    # Cluster centers (in original feature scale if standardized)
    if scale_features:
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=use_cols)
    else:
        centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=use_cols)
    centers_df.index.name = "Cluster"
    st.subheader("Cluster centers (feature means per cluster)")
    st.dataframe(centers_df)

    # Cluster-wise means
    st.subheader("Cluster-wise means (dataset rows assigned to clusters)")
    cluster_means = df_clustered.groupby("Cluster")[use_cols].mean()
    st.dataframe(cluster_means)

    # PCA for visualization
    pca = PCA(n_components=3 if show_3d else 2, random_state=int(random_state))
    pca_trans = pca.fit_transform(X_scaled)
    df_clustered["PCA1"] = pca_trans[:, 0]
    df_clustered["PCA2"] = pca_trans[:, 1]
    if show_3d:
        if pca_trans.shape[1] < 3:
            st.info("3D PCA not available; less than 3 PC components.")
        else:
            df_clustered["PCA3"] = pca_trans[:, 2]

    # 2D PCA scatter
    st.subheader("PCA — 2D cluster scatter")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_clustered, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=60, ax=ax)
    ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
    ax.set_title("PCA (2 components) — clusters")
    st.pyplot(fig)

    # optional 3D
    if show_3d and "PCA3" in df_clustered.columns:
        st.subheader("PCA — 3D cluster scatter")
        fig3 = plt.figure(figsize=(8,6))
        ax3 = fig3.add_subplot(111, projection='3d')
        for c in sorted(df_clustered["Cluster"].unique()):
            subset = df_clustered[df_clustered["Cluster"]==c]
            ax3.scatter(subset["PCA1"], subset["PCA2"], subset["PCA3"], label=f"Cluster {c}", s=20)
        ax3.set_xlabel("PCA1"); ax3.set_ylabel("PCA2"); ax3.set_zlabel("PCA3")
        ax3.legend()
        st.pyplot(fig3)

    # Elbow & silhouette across k (optional)
    if run_elbow:
        st.subheader("Elbow & Silhouette analysis")
        max_k = min(12, max(6, int(min(20, len(X_scaled) / 5))))  # heuristic to not try too many k on small datasets
        inertias = []
        sils = []
        tried = range(2, max_k + 1)
        for kk in tried:
            km = KMeans(n_clusters=kk, random_state=int(random_state), n_init=5)
            lab = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            try:
                s = silhouette_score(X_scaled, lab) if len(set(lab)) > 1 else np.nan
            except:
                s = np.nan
            sils.append(s)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(list(tried), inertias, marker='o')
        ax[0].set_title("Elbow Method (Inertia)")
        ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
        ax[1].plot(list(tried), sils, marker='o')
        ax[1].set_title("Silhouette Score vs k")
        ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
        st.pyplot(fig)

with tab_plots:
    st.subheader("All visualizations (many charts)")

    # Histogram grid
    st.markdown("### Histograms (feature distributions)")
    cols = use_cols
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        sns.histplot(df[c].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(c)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplots
    st.markdown("### Boxplots (outliers & spread)")
    fig, ax = plt.subplots(figsize=(max(8, len(cols)), 5))
    sns.boxplot(data=df[cols], orient="h", ax=ax)
    st.pyplot(fig)

    # Violin plots per feature grouped by cluster (if clustered)
    if "Cluster" in df.columns or "Cluster" in df_clustered.columns:
        st.markdown("### Per-feature violin plots by cluster")
        # ensure df_clustered exists
        try:
            display_df = df_clustered.copy()
        except NameError:
            display_df = df.copy()
            display_df["Cluster"] = kmeans.predict(X_scaled)
        # limit number of features to visualize to avoid huge output
        for c in cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.violinplot(x="Cluster", y=c, data=display_df, inner="quartile", ax=ax)
            ax.set_title(f"{c} by Cluster")
            st.pyplot(fig)

    # Correlation heatmap
    st.markdown("### Correlation matrix (heatmap)")
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(min(12, len(cols)), min(10, len(cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot (subsample if needed)
    st.markdown("### Pairplot (scatter matrix) — may take time for many features")
    # decide sample
    if sample_for_pairplot and sample_for_pairplot > 0 and sample_for_pairplot < len(df):
        pp_df = df.loc[X.index, cols].sample(n=sample_for_pairplot, random_state=int(random_state))
        # if cluster info exists, add it
        if "Cluster" in df_clustered.columns:
            pp_df = pp_df.join(df_clustered["Cluster"], how="left")
    else:
        pp_df = df.loc[X.index, cols].copy()
        if "Cluster" in df_clustered.columns:
            pp_df = pp_df.join(df_clustered["Cluster"], how="left")

    # limit pairplot columns to max 6 to avoid overload
    max_pairplot_cols = 6
    pairplot_cols = pp_df.columns.tolist()
    if "Cluster" in pairplot_cols:
        pairplot_cols = pairplot_cols  # cluster included
    if len([c for c in pairplot_cols if c != "Cluster"]) > max_pairplot_cols:
        # pick top features by variance
        feats = [c for c in pairplot_cols if c != "Cluster"]
        variances = df[feats].var().sort_values(ascending=False)
        keep = variances.index[:max_pairplot_cols].tolist()
        if "Cluster" in pp_df.columns:
            keep.append("Cluster")
        pp_df = pp_df[keep]
        st.info(f"Pairplot limited to top {max_pairplot_cols} numeric features by variance: {keep}")

    try:
        pairplot_fig = sns.pairplot(pp_df, hue="Cluster" if "Cluster" in pp_df.columns else None, corner=True)
        st.pyplot(pairplot_fig)
    except Exception as e:
        st.warning(f"Pairplot failed or is too heavy to render here: {e}")

with tab_stats:
    st.subheader("Export & quick summary")
    
    st.markdown("### Cluster summary table (counts & means)")

    
    try:
        summary = df_clustered.groupby("Cluster")[use_cols].agg(["count", "mean"]).round(3)
        st.dataframe(summary)
    except Exception:
        st.warning("No clustered dataframe available to show summary table.")

    st.markdown("### Download clustered data")
    try:
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV of clustered data", data=csv, file_name="clustered_customers.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"Unable to prepare download: {e}")


        

    st.markdown("## Cluster Profiling Interpretation")

    st.write("""
    **Cluster 0 — Low Engagement, Mid-Income Professionals:**  
    Customers in this cluster are middle-aged with average annual income but low spending scores.  
    They tend to make purchases occasionally and are more price or need conscious than brand oriented.
    """)

    st.write("""
    **Cluster 1 — High Income, Active Premium Spenders:**  
    These customers are financially strong with a moderate-high spending score.  
    They are more open to branded or premium product offerings and represent a high-value segment.
    """)

    st.write("""
    **Cluster 2 — Loyal Value Seekers (High Spending but Low Income):**  
    Although these customers have lower income, they show high spending behavior and longer tenure.  
    They are loyal emotional buyers influenced by offers, brand relationships, and festive promotions.
    """)
    # ---------------- Marketing Recommendations Section ----------------
    st.markdown("## 🎯 Marketing & Business Strategy Suggestions")

    st.write("""
    **For Cluster 0:**  
    • Offer cashback and loyalty reward programs  
    • Send targeted discount notifications  
    • Recommend products based on browsing/purchase history  
    """)

    st.write("""
    **For Cluster 1:**  
    • Introduce VIP / Premium Membership Programs  
    • Give early access to new product launches  
    • Use premium & aspirational brand messaging  
    """)
    st.write("""
    **For Cluster 2:**  
    • Provide bundle discounts and festival sale offers  
    • Strengthen relationship marketing through personalized messages  
    • Reward repeat purchases with loyalty points  
    """)