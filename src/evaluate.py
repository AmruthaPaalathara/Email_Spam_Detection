import joblib, numpy as np, matplotlib.pyplot as plt # for loading saved model artifacts,  numerical operations, plotting confusion matrices and ROC curves

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

# our preprocessing helper: loads & cleans text, returns X (TF-IDF), y (labels), and the vectorizer
from src.preprocess import load_and_preprocess

def plot_cm(cm, title):
    
    """
    Display a confusion matrix `cm` with a color map and proper axis labels.
    """
    
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(title); plt.colorbar()
    plt.xticks([0,1], ['ham','spam']); plt.yticks([0,1], ['ham','spam'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()

def main():
    X, y, _ = load_and_preprocess()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    for name, fname in [('NB','nb.joblib'), ('SVM','svm.joblib')]:
         # 3) Iterate over both models: Naive Bayes (NB) and Support Vector Machine (SVM)
        m = joblib.load(f"models/{fname}")
        ypred = m.predict(Xte)
        acc = accuracy_score(yte, ypred)
        p, r, f1, _ = precision_recall_fscore_support(
            yte, ypred, average='binary', pos_label='spam'
        )
        print(f"{name}: acc={acc:.3f}, p={p:.3f}, r={r:.3f}, f1={f1:.3f}")

        cm = confusion_matrix(yte, ypred, labels=['ham','spam'])
        plot_cm(cm, f"{name} Confusion Matrix")

        # ROC
        if hasattr(m, 'decision_function'):
            scores = m.decision_function(Xte)
        else:
            scores = m.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve((yte=='spam').astype(int), scores)
        plt.figure(); plt.plot(fpr, tpr)
        plt.title(f"{name} ROC Curve")
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
