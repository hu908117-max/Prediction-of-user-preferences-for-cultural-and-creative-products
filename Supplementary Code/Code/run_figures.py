"""Generate all result figures (Figs 2-8) from the manuscript result code.

This is the Python counterpart of the original result-figure code. It produces
the publication-quality figures under outputs/figures/.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "src", "figures")
OUT_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, FIG_DIR)

import figure2_tsne
import figure3_infonce
import figure4_loss
import figure5_classification
import figure6_ranking
import figure7_roc
import figure8_sensitivity


def main():
    figure2_tsne.build_figure2(os.path.join(OUT_DIR, "Fig_UserPreference_tSNE_Compare.png"))
    figure3_infonce.build_figure3(os.path.join(OUT_DIR, "Fig_InfoNCE_Similarity_Distribution.png"))
    figure4_loss.build_figure4(os.path.join(OUT_DIR, "Fig_JointLoss_Convergence.png"))
    figure5_classification.build_figure5(os.path.join(OUT_DIR, "Fig_Classification_Comparison.png"))
    figure6_ranking.build_figure6(os.path.join(OUT_DIR, "Fig_TopK_Ranking_Comparison.png"))
    figure7_roc.build_figure7(os.path.join(OUT_DIR, "Fig_ROC_ConfidenceBand.png"))
    figure8_sensitivity.build_figure8(os.path.join(OUT_DIR, "Fig_Parameter_Sensitivity.png"))
    print("All figures generated under:", OUT_DIR)


if __name__ == "__main__":
    main()
