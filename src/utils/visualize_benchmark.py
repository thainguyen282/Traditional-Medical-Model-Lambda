
import os 
import json
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd
from pathlib import Path

#  Assume that all the ckp eval on the same set of subtask
def chart_benchmark(model_ckp_map:dict,tasks:list):
    for task in tasks:
        for model_tag, root_folder in model_ckp_map.items():
            visualize_single_model_eval_ckp(model_tag,root_folder,task,output_dir=root_folder)
         
def visualize_single_model_eval_ckp(
    model_tag: str,
    root_folder: str,
    task: str,
    output_dir: str = "visualizations",
    save_format: str = "png",  # can be "png", "pdf", "svg", etc.
    dpi: int = 300,           # resolution for raster formats
    show: bool = True          # whether to display the plot
):
    """
    Visualize model performance and save to output directory.
    
    Args:
        model_tag: Prefix for model names
        root_folder: Directory containing checkpoint folders
        task: Task name (subfolder under each checkpoint)
        subtasks: List of subtasks to evaluate
        output_dir: Directory to save visualizations
        save_format: Image format (png, pdf, svg, etc.)
        dpi: Resolution in dots per inch
        show: Whether to display the plot interactively
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ckp_subfolders = [item for item in os.listdir(root_folder) 
                     if os.path.isdir(os.path.join(root_folder, item))]
    assert ckp_subfolders, f"No checkpoints found in {root_folder}"
    
    data = []
    discovered_subtasks = set()
    
    for ckp in ckp_subfolders:
        ckp_path = os.path.join(root_folder, ckp, task)
        for file in os.listdir(ckp_path):
            if file.endswith(".json"):
                with open(os.path.join(ckp_path, file)) as f:
                    results = json.load(f)
                
                if not discovered_subtasks:
                    discovered_subtasks = set(results["results"].keys())
                
                for subtask in discovered_subtasks:
                    data.append({
                        'Model': f"{model_tag}_{ckp}",
                        'Subtask': subtask,
                        'Accuracy': results["results"][subtask]["acc,none"]
                    })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(max(12, len(discovered_subtasks)), 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        x="Subtask",
        y="Accuracy",
        hue="Model",
        data=df,
        palette="viridis",
        dodge=True,
        errorbar=None,
        saturation=0.8
    )
    
    rotation = 45 if len(discovered_subtasks) > 5 else 0
    title = f'Model Performance on {task} Subtasks'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Subtask', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=rotation, ha='right' if rotation else 'center')
    
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        ncol=1 if len(ckp_subfolders) < 5 else 2
    )
    
    plt.tight_layout()
    
    filename = f"{model_tag}_{task}_performance.{save_format}"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    

           
