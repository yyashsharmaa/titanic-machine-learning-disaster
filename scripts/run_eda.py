"""
Exploratory Data Analysis (EDA) script for Titanic dataset.
Generates publication-quality visualizations answering:
"What sorts of people were more likely to survive?"
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.features import extract_title, extract_cabin_deck

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def run_eda():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    train_df, _ = load_raw_data()
    
    # Enrich for analysis
    train_df["Title"] = train_df["Name"].apply(extract_title)
    train_df["CabinDeck"] = train_df["Cabin"].apply(extract_cabin_deck)
    train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
    train_df["IsAlone"] = (train_df["FamilySize"] == 1).astype(int)
    
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {0: "#e11d48", 1: "#059669"} # Crimson vs Emerald
    
    # 1. Survival by Sex and Pclass
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=train_df, x="Pclass", y="Survived", hue="Sex",
        palette={"male": "#3b82f6", "female": "#ec4899"}, errorbar=None
    )
    plt.title("Titanic Survival Rate by Passenger Class & Gender", fontsize=14, weight="bold", pad=15)
    plt.ylabel("Survival Rate", fontsize=12)
    plt.xlabel("Passenger Class (1st, 2nd, 3rd)", fontsize=12)
    plt.ylim(0, 1.05)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1%}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, weight="bold",
                        xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "eda_survival_by_gender_class.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Generated {p1}")

    # 2. Age Distribution by Survival Status
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=train_df[train_df["Survived"] == 1]["Age"], label="Survived (1)", color="#059669", fill=True, alpha=0.4)
    sns.kdeplot(data=train_df[train_df["Survived"] == 0]["Age"], label="Did Not Survive (0)", color="#e11d48", fill=True, alpha=0.4)
    plt.axvline(train_df["Age"].median(), color="#475569", linestyle="--", label=f"Median Age ({train_df['Age'].median():.0f})")
    plt.title("Age Distribution by Survival Status (KDE)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper right")
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "eda_age_distribution.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Generated {p2}")

    # 3. Family Size & Survival
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=train_df, x="FamilySize", y="Survived",
        color="#6366f1", errorbar=None
    )
    plt.title("Survival Rate vs. Family Size (SibSp + Parch + 1)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Family Size (Passengers on Same Ticket/Family)", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    plt.ylim(0, 0.9)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1%}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, weight="bold",
                        xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "eda_family_size_impact.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Generated {p3}")

    # 4. Cabin Deck & Survival
    plt.figure(figsize=(9, 5))
    deck_order = ["A", "B", "C", "D", "E", "F", "G", "U"]
    ax = sns.barplot(
        data=train_df, x="CabinDeck", y="Survived",
        order=deck_order, palette="mako", errorbar=None
    )
    plt.title("Survival Rate by Cabin Deck (U = Unknown/No Cabin Recorded)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Cabin Deck", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    plt.ylim(0, 1.05)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1%}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, weight="bold",
                        xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    p4 = os.path.join(FIGURES_DIR, "eda_cabin_deck_impact.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Generated {p4}")

    # 5. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    corr = train_df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    p5 = os.path.join(FIGURES_DIR, "eda_correlation_heatmap.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Generated {p5}")
    
    print("\nEDA Visualizations successfully generated!")


if __name__ == "__main__":
    run_eda()
