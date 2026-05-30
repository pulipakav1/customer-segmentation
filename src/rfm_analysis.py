import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def load_data():
    customers = pd.read_csv('data/customers.csv')
    payments   = pd.read_csv('data/payments.csv')
    payments['payment_date'] = pd.to_datetime(payments['payment_date'], utc=True)
    payments = payments[payments['payment_status'] == 'Success']
    return customers, payments


def compute_rfm(customers, payments):
    snapshot_date = payments['payment_date'].max() + pd.Timedelta(days=1)
    rfm = payments.groupby('customer_id').agg(
        recency   = ('payment_date', lambda x: (snapshot_date - x.max()).days),
        frequency = ('payment_id', 'count'),
        monetary  = ('amount', 'sum')
    ).reset_index()
    rfm = rfm.merge(customers[['customer_id', 'segment', 'acquisition_channel']], on='customer_id', how='left')
    return rfm


def find_optimal_k(rfm_scaled: np.ndarray, k_range=range(2, 9)):
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Cluster Validation — Choosing Optimal k', fontsize=12, fontweight='bold')

    ks = list(k_range)
    axes[0].plot(ks, inertias, marker='o', color='#4472C4', linewidth=2)
    axes[0].set_title('Elbow Method (Inertia)')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, sil_scores, marker='o', color='#2ecc71', linewidth=2)
    best_k = ks[int(np.argmax(sil_scores))]
    axes[1].axvline(best_k, color='#e74c3c', linestyle='--', label=f'Best k={best_k}')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/cluster_validation.png', dpi=150, bbox_inches='tight')
    print(f"Saved reports/cluster_validation.png  (best k by silhouette: {best_k})")

    return best_k, inertias, sil_scores


def label_clusters_from_centroids(cluster_summary: pd.DataFrame) -> pd.Series:
    """Assign labels by ranking cluster centroids — no hardcoded thresholds."""
    df = cluster_summary.copy()
    # Higher recency = older last purchase = worse; invert rank
    df['r_rank'] = df['avg_recency'].rank(ascending=True)   # lower recency = better
    df['f_rank'] = df['avg_frequency'].rank(ascending=False)
    df['m_rank'] = df['avg_monetary'].rank(ascending=False)
    df['score'] = df['r_rank'] + df['f_rank'] + df['m_rank']
    df = df.sort_values('score')

    n = len(df)
    labels = ['Champions', 'Loyal', 'At Risk', 'Lost'][:n]
    label_map = {}
    for i, idx in enumerate(df.index):
        label_map[idx] = labels[i] if i < len(labels) else f'Segment {i+1}'
    return pd.Series(label_map)


def compute_ltv(rfm: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """Estimate LTV per segment: avg monthly spend × expected active months."""
    payments_with_label = payments.merge(rfm[['customer_id', 'segment_label']], on='customer_id', how='left')
    payments_with_label['month'] = payments_with_label['payment_date'].dt.to_period('M')

    monthly_spend = (
        payments_with_label
        .groupby(['customer_id', 'month'])['amount'].sum()
        .reset_index()
        .groupby('customer_id')['amount'].mean()
        .reset_index()
        .rename(columns={'amount': 'avg_monthly_spend'})
    )
    rfm_ltv = rfm.merge(monthly_spend, on='customer_id', how='left')

    tenure_months = (rfm_ltv['recency'].max() - rfm_ltv['recency']) / 30
    rfm_ltv['active_months'] = tenure_months.clip(lower=1)

    ltv_summary = (
        rfm_ltv.groupby('segment_label')
        .agg(
            customers         = ('customer_id', 'count'),
            avg_monthly_spend = ('avg_monthly_spend', 'mean'),
            avg_active_months = ('active_months', 'mean'),
        )
        .round(2)
    )
    ltv_summary['estimated_ltv'] = (
        ltv_summary['avg_monthly_spend'] * ltv_summary['avg_active_months']
    ).round(2)
    return ltv_summary.sort_values('estimated_ltv', ascending=False)


def main():
    customers, payments = load_data()
    rfm = compute_rfm(customers, payments)

    print(f"customers with RFM scores : {len(rfm):,}")
    print(f"avg recency (days)        : {rfm['recency'].mean():.0f}")
    print(f"avg frequency (payments)  : {rfm['frequency'].mean():.1f}")
    print(f"avg monetary value        : ${rfm['monetary'].mean():.2f}")
    print(f"total revenue             : ${rfm['monetary'].sum():,.2f}")

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

    best_k, _, sil_scores = find_optimal_k(rfm_scaled)
    k = best_k if best_k in range(3, 6) else 4

    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm['cluster'] = km_final.fit_predict(rfm_scaled)

    final_sil = silhouette_score(rfm_scaled, rfm['cluster'])
    print(f"\nFinal model: k={k}, silhouette score={final_sil:.3f}")

    cluster_summary = rfm.groupby('cluster').agg(
        customers     = ('customer_id', 'count'),
        avg_recency   = ('recency', 'mean'),
        avg_frequency = ('frequency', 'mean'),
        avg_monetary  = ('monetary', 'mean'),
        total_revenue = ('monetary', 'sum')
    ).round(2)
    cluster_summary['revenue_pct'] = (
        cluster_summary['total_revenue'] / cluster_summary['total_revenue'].sum() * 100
    ).round(1)

    label_map = label_clusters_from_centroids(cluster_summary)
    cluster_summary['label'] = label_map

    print("\n--- cluster summary ---")
    print(cluster_summary.to_string())

    rfm['segment_label'] = rfm['cluster'].map(cluster_summary['label'])

    # LTV per segment
    ltv_summary = compute_ltv(rfm, payments)
    print("\n--- estimated LTV by segment ---")
    print(ltv_summary.to_string())

    # Plots
    colors = {'Champions': '#2ecc71', 'Loyal': '#3498db', 'At Risk': '#e67e22', 'Lost': '#e74c3c'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Customer Segmentation — RFM Analysis', fontsize=14, fontweight='bold')

    # Segment counts
    ax1 = axes[0, 0]
    seg_counts = rfm['segment_label'].value_counts()
    bars = ax1.bar(seg_counts.index, seg_counts.values,
                   color=[colors.get(s, '#95a5a6') for s in seg_counts.index])
    ax1.set_title('Customer Count by Segment')
    ax1.set_ylabel('Customers')
    for bar, val in zip(bars, seg_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Revenue by segment
    ax2 = axes[0, 1]
    seg_revenue = rfm.groupby('segment_label')['monetary'].sum().sort_values(ascending=False)
    bars2 = ax2.bar(seg_revenue.index, seg_revenue.values,
                    color=[colors.get(s, '#95a5a6') for s in seg_revenue.index])
    ax2.set_title('Total Revenue by Segment')
    ax2.set_ylabel('Revenue ($)')
    for bar, val in zip(bars2, seg_revenue.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                 f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Estimated LTV
    ax3 = axes[0, 2]
    ltv_plot = ltv_summary['estimated_ltv'].sort_values(ascending=False)
    bars3 = ax3.bar(ltv_plot.index, ltv_plot.values,
                    color=[colors.get(s, '#95a5a6') for s in ltv_plot.index])
    ax3.set_title('Estimated LTV by Segment')
    ax3.set_ylabel('Estimated LTV ($)')
    for bar, val in zip(bars3, ltv_plot.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Recency vs Monetary
    ax4 = axes[1, 0]
    for label, group in rfm.groupby('segment_label'):
        ax4.scatter(group['recency'], group['monetary'],
                    alpha=0.3, label=label, color=colors.get(label, '#95a5a6'), s=10)
    ax4.set_xlabel('Recency (days)')
    ax4.set_ylabel('Monetary ($)')
    ax4.set_title('Recency vs Monetary')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Frequency vs Monetary
    ax5 = axes[1, 1]
    for label, group in rfm.groupby('segment_label'):
        ax5.scatter(group['frequency'], group['monetary'],
                    alpha=0.3, label=label, color=colors.get(label, '#95a5a6'), s=10)
    ax5.set_xlabel('Frequency (payments)')
    ax5.set_ylabel('Monetary ($)')
    ax5.set_title('Frequency vs Monetary')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # RFM heatmap — segment profiles
    ax6 = axes[1, 2]
    profile = cluster_summary[['avg_recency', 'avg_frequency', 'avg_monetary']].copy()
    profile.index = cluster_summary['label']
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
    profile_norm['avg_recency'] = 1 - profile_norm['avg_recency']  # invert recency
    im = ax6.imshow(profile_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax6.set_xticks(range(3))
    ax6.set_xticklabels(['Recency\n(inverted)', 'Frequency', 'Monetary'])
    ax6.set_yticks(range(len(profile_norm)))
    ax6.set_yticklabels(profile_norm.index)
    ax6.set_title('Segment RFM Profile')
    for i in range(len(profile_norm)):
        for j in range(3):
            ax6.text(j, i, f'{profile_norm.values[i, j]:.2f}',
                     ha='center', va='center', fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax6, shrink=0.8)

    plt.tight_layout()
    plt.savefig('reports/rfm_segments.png', dpi=150, bbox_inches='tight')
    print("\nplot saved to reports/rfm_segments.png")

    rfm.to_csv('reports/rfm_scores.csv', index=False)
    ltv_summary.to_csv('reports/ltv_by_segment.csv')
    print("rfm scores saved to reports/rfm_scores.csv")
    print("ltv summary saved to reports/ltv_by_segment.csv")


if __name__ == '__main__':
    main()
