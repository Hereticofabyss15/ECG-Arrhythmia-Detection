"""Evaluate saved predictions on untouched DS2 patients."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

MODEL_DIR, RESULTS_DIR = "models", "results"
CLASSES = ["N", "SVEB", "VEB", "F", "Q"]

def evaluate(y_true, y_pred, name):
    cm=confusion_matrix(y_true,y_pred,labels=CLASSES)
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=CLASSES,zero_division=0)
    pc=pd.DataFrame({"class":CLASSES,"precision":p,"recall_sensitivity":r,"f1":f,"support":s})
    pc["specificity"]=[(cm.sum()-cm[i].sum()-cm[:,i].sum()+cm[i,i])/max(cm.sum()-cm[i].sum(),1) for i in range(len(CLASSES))]
    pc.to_csv(os.path.join(RESULTS_DIR,name.lower()+"_per_class_metrics.csv"),index=False)
    plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=CLASSES,yticklabels=CLASSES); plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(name+" — DS2 confusion matrix"); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,name.lower()+"_confusion_matrix.png"),dpi=300); plt.close()
    result={"Model":name,"Accuracy":accuracy_score(y_true,y_pred),"Balanced Accuracy":balanced_accuracy_score(y_true,y_pred),"Macro F1":f1_score(y_true,y_pred,labels=CLASSES,average="macro",zero_division=0),"Macro Precision":p.mean(),"Macro Recall / Sensitivity":r.mean(),"Macro Specificity":pc.specificity.mean()}
    print("\n"+name+"\n"+classification_report(y_true,y_pred,labels=CLASSES,zero_division=0)); print(pc.to_string(index=False)); return result

def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    df=pd.read_csv(os.path.join(MODEL_DIR,"test_predictions.csv")); y=df.label
    cols=[c for c in df if c.endswith("_prediction")]
    if not cols: raise ValueError("No prediction columns found; run train_models.py first.")
    comparison=pd.DataFrame([evaluate(y,df[c],c.removesuffix("_prediction")) for c in cols]).sort_values("Macro F1",ascending=False)
    comparison.to_csv(os.path.join(RESULTS_DIR,"model_comparison.csv"),index=False)
    plt.figure(figsize=(8,5)); plt.bar(comparison.Model,comparison["Macro F1"]); plt.ylim(0,1); plt.ylabel("DS2 Macro F1"); plt.title("Patient-independent model comparison"); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"model_f1_comparison.png"),dpi=300); plt.close()
    print("\nDS2 model comparison:\n",comparison.to_string(index=False))
if __name__ == "__main__": main()
