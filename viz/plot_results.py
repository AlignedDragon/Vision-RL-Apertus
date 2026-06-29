import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEIL = 41.4  # base@2048 direct
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

labelsA = ["base\n(direct)", "SFT\nno-zoom", "SFT\n+zoom", "RL\n+zoom", "RL\nno-zoom"]
valsA = [33.0, 32.5, 35.6, 32.5, 26.7]
colA = ["#888888", "#7aa6c2", "#1f6f9e", "#c2785a", "#e0b0a0"]
b = ax1.bar(labelsA, valsA, color=colA)
ax1.axhline(CEIL, ls="--", c="k", lw=1)
ax1.text(4.45, CEIL + 0.3, f"base@2048 ceiling {CEIL}%", ha="right", fontsize=9)
ax1.set_title("V*Bench @ 256-token image budget", fontsize=12, fontweight="bold")
ax1.set_ylabel("accuracy (%)"); ax1.set_ylim(0, 48)
for r, v in zip(b, valsA):
    ax1.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v}", ha="center", fontsize=9)

labelsB = ["base@256\ndirect", "base@2048\ndirect", "SFT@2048\n+zoom", "RL@2048\n+zoom"]
valsB = [33.0, 41.4, 37.7, 44.0]
colB = ["#bbbbbb", "#888888", "#1f6f9e", "#c2785a"]
b2 = ax2.bar(labelsB, valsB, color=colB)
ax2.set_title("Resolution ceiling & 2048-input zoom", fontsize=12, fontweight="bold")
ax2.set_ylabel("accuracy (%)"); ax2.set_ylim(0, 48)
for r, v in zip(b2, valsB):
    ax2.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v}", ha="center", fontsize=9)

fig.suptitle("Chain-of-Focus on Apertus-8B @ 256-token budget (V*Bench, sglang greedy)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("viz/vstar_256_summary.png", dpi=130)
print("saved viz/vstar_256_summary.png")
