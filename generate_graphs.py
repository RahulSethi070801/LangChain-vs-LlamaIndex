import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

indices = range(len(df))

# ----------- Plot 1: Time Comparison -----------
plt.figure(figsize=(12, 4))
plt.plot(df["LangChain Time (s)"], label="LangChain", marker='o')
plt.plot(df["LlamaIndex Time (s)"], label="LlamaIndex", marker='x')
plt.title("Query Response Time Comparison")
plt.ylabel("Time (s)")
plt.xlabel("Query Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------- Plot 2: F1 Score Comparison -----------
plt.figure(figsize=(12, 4))
plt.plot(df["LangChain F1"], label="LangChain", marker='o')
plt.plot(df["LlamaIndex F1"], label="LlamaIndex", marker='x')
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xlabel("Query Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------- Plot 3: Manual Validation Comparison -----------
bar_width = 0.35

plt.figure(figsize=(12, 4))
plt.bar([i - bar_width/2 for i in indices], df["LangChain Manual Validation"], 
        width=bar_width, label="LangChain", color='skyblue')
plt.bar([i + bar_width/2 for i in indices], df["LlamaIndex Manual Validation"], 
        width=bar_width, label="LlamaIndex", color='orange')

plt.title("Manual Validation (Correct Answers)")
plt.ylabel("Validated as Correct (1 = Yes, 0 = No)")
plt.xlabel("Query Index")
plt.xticks(indices)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
