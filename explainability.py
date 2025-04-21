import pandas as pd

# Define columns
columns = ["protein_id", "category", "pred_spearman", "true_spearman"]

# Load ProtLigand and SaProt data
protligand_df = pd.read_csv("output/Thermostability/protligand_cross_attention.tsv", sep="\t", names=columns)
saprot_df = pd.read_csv("output/Thermostability/SaProt_650M_AF2.tsv", sep="\t", names=columns)

# Merge on protein ID
merged_df = pd.merge(
    protligand_df,
    saprot_df,
    on="protein_id",
    suffixes=("_protligand", "_saprot")
)

# merged_df.to_csv("output/Thermostability/merged_results.csv")

# Compute the absolute difference between prediction and true spearman
merged_df["error_protligand"] = (merged_df["pred_spearman_protligand"] - merged_df["true_spearman_protligand"]).abs()
merged_df["error_saprot"] = (merged_df["pred_spearman_saprot"] - merged_df["true_spearman_saprot"]).abs()
merged_df["delta"] = merged_df["error_saprot"] - merged_df["error_protligand"]  # positive: ProtLigand better
merged_df["abs_delta"] = merged_df["delta"].abs()

# Count proteins per category
category_counts = merged_df["category_protligand"].value_counts()
print("Number of proteins per category:\n", category_counts, "\n")

# Performance delta per category
category_stats = merged_df.groupby("category_protligand").agg({
    "delta": ["mean", "std", "count"],
    "error_protligand": "mean",
    "error_saprot": "mean"
}).sort_values(("delta", "mean"), ascending=False)

print("Performance gap per category (positive = ProtLigand better):\n", category_stats, "\n")

# Top 10 proteins where ProtLigand did much better
top_improved = merged_df.sort_values("delta", ascending=False).head(10)
print("Top 10 proteins where ProtLigand was significantly more accurate:\n", top_improved[[
    "protein_id", "category_protligand", "error_protligand", "error_saprot", "delta"
]], "\n")

# Top 10 proteins where SaProt did better
top_degraded = merged_df.sort_values("delta", ascending=True).head(10)
print("Top 10 proteins where SaProt was more accurate:\n", top_degraded[[
    "protein_id", "category_protligand", "error_protligand", "error_saprot", "delta"
]], "\n")
