import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

customers = pd.read_csv('data/customers.csv')
payments   = pd.read_csv('data/payments.csv')

payments['payment_date'] = pd.to_datetime(payments['payment_date'], utc=True)
payments = payments[payments['payment_status'] == 'Success']

snapshot_date = payments['payment_date'].max() + pd.Timedelta(days=1)

rfm = payments.groupby('customer_id').agg(
    recency   = ('payment_date', lambda x: (snapshot_date - x.max()).days),
    frequency = ('payment_id', 'count'),
    monetary  = ('amount', 'sum')
).reset_index()

rfm = rfm.merge(customers[['customer_id', 'segment', 'acquisition_channel']], on='customer_id', how='left')

print(f"customers with RFM scores : {len(rfm):,}")
print(f"avg recency (days)        : {rfm['recency'].mean():.0f}")
print(f"avg frequency (payments)  : {rfm['frequency'].mean():.1f}")
print(f"avg monetary value        : ${rfm['monetary'].mean():.2f}")
print(f"total revenue             : ${rfm['monetary'].sum():,.2f}")

scaler   = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['cluster'] = km_final.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby('cluster').agg(
    customers  = ('customer_id', 'count'),
    avg_recency   = ('recency', 'mean'),
    avg_frequency = ('frequency', 'mean'),
    avg_monetary  = ('monetary', 'mean'),
    total_revenue = ('monetary', 'sum')
).round(2)

cluster_summary['revenue_pct'] = (cluster_summary['total_revenue'] / cluster_summary['total_revenue'].sum() * 100).round(1)
print("\n--- cluster summary ---")
print(cluster_summary.to_string())

def label_cluster(row):
    if row['avg_recency'] > 400:
        return 'Lost'
    elif row['avg_frequency'] > 20 and row['avg_monetary'] > 2000:
        return 'Champions'
    elif row['avg_frequency'] > 20 and row['avg_monetary'] > 1000:
        return 'Loyal'
    else:
        return 'At Risk'

cluster_summary['label'] = cluster_summary.apply(label_cluster, axis=1)
print("\n--- cluster labels ---")
print(cluster_summary[['customers', 'avg_recency', 'avg_frequency', 'avg_monetary', 'total_revenue', 'revenue_pct', 'label']].to_string())

rfm['segment_label'] = rfm['cluster'].map(cluster_summary['label'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Segmentation — RFM Analysis', fontsize=14, fontweight='bold')

colors = {'Champions': '#2ecc71', 'Loyal': '#3498db', 'At Risk': '#e67e22', 'Lost': '#e74c3c'}

ax1 = axes[0, 0]
seg_counts = rfm['segment_label'].value_counts()
bars = ax1.bar(seg_counts.index, seg_counts.values,
               color=[colors.get(s, '#95a5a6') for s in seg_counts.index])
ax1.set_title('Customer Count by Segment')
ax1.set_ylabel('Customers')
for bar, val in zip(bars, seg_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:,}', ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[0, 1]
seg_revenue = rfm.groupby('segment_label')['monetary'].sum().sort_values(ascending=False)
bars2 = ax2.bar(seg_revenue.index, seg_revenue.values,
                color=[colors.get(s, '#95a5a6') for s in seg_revenue.index])
ax2.set_title('Total Revenue by Segment')
ax2.set_ylabel('Revenue ($)')
for bar, val in zip(bars2, seg_revenue.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

ax3 = axes[1, 0]
for label, group in rfm.groupby('segment_label'):
    ax3.scatter(group['recency'], group['monetary'],
                alpha=0.3, label=label, color=colors.get(label, '#95a5a6'), s=10)
ax3.set_xlabel('Recency (days since last payment)')
ax3.set_ylabel('Monetary Value ($)')
ax3.set_title('Recency vs Monetary Value')
ax3.legend()
ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
for label, group in rfm.groupby('segment_label'):
    ax4.scatter(group['frequency'], group['monetary'],
                alpha=0.3, label=label, color=colors.get(label, '#95a5a6'), s=10)
ax4.set_xlabel('Frequency (number of payments)')
ax4.set_ylabel('Monetary Value ($)')
ax4.set_title('Frequency vs Monetary Value')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/rfm_segments.png', dpi=150, bbox_inches='tight')
print("\nplot saved to reports/rfm_segments.png")

rfm.to_csv('reports/rfm_scores.csv', index=False)
print("rfm scores saved to reports/rfm_scores.csv")