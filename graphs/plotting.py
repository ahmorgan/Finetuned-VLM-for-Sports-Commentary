import re
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_and_eval_logs(log_file_path, output_filename):
    with open(log_file_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # 1. Parse Training Metrics
    train_pattern = r"\{'loss':.*?\}|\{.*?'loss':.*?\}"
    train_matches = re.findall(train_pattern, log_content)
    
    train_data = []
    for match in train_matches:
        if "'eval_loss'" in match: 
            continue # Skip evaluation dictionaries
        try:
            log_dict = ast.literal_eval(match)
            parsed_dict = {k: float(v) for k, v in log_dict.items() if v != 'nan'}
            train_data.append(parsed_dict)
        except (ValueError, SyntaxError):
            continue
            
    df_train = pd.DataFrame(train_data)

    # 2. Parse Evaluation Metrics
    eval_iter = re.finditer(r"\{'eval_loss':.*?\}", log_content)
    eval_data = []
    eval_epochs = []
    
    for match in eval_iter:
        try:
            d = ast.literal_eval(match.group())
            parsed_dict = {k: float(v) for k, v in d.items() if v != 'nan'}
            eval_data.append(parsed_dict)
            eval_epochs.append((match.end(), parsed_dict.get('epoch', np.nan)))
        except (ValueError, SyntaxError):
            continue
            
    df_eval = pd.DataFrame(eval_data)
    
    # 3. Parse ROUGE Metrics
    rouge_iter = re.finditer(r"ROUGE score: (\{.*?\})", log_content)
    rouge_data = []
    for match in rouge_iter:
        pos = match.start()
        r_str = match.group(1)
        # Strip numpy wrappers for ast.literal_eval
        r_str_clean = re.sub(r"np\.float64\((.*?)\)", r"\1", r_str)
        try:
            r_dict = ast.literal_eval(r_str_clean)
            
            # Map ROUGE score to the closest preceding evaluation epoch
            current_epoch = np.nan
            for eval_pos, epoch in reversed(eval_epochs):
                if eval_pos < pos:
                    current_epoch = epoch
                    break
                    
            if not np.isnan(current_epoch):
                r_dict['epoch'] = current_epoch
                rouge_data.append(r_dict)
        except (ValueError, SyntaxError):
            pass

    has_rouge = False
    if rouge_data:
        df_rouge = pd.DataFrame(rouge_data)
        df_rouge_avg = df_rouge.groupby('epoch').mean().reset_index()
        has_rouge = not df_rouge_avg.empty
    
    # Determine plot structure
    num_subplots = 4 if has_rouge else 3
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))
    
    # Plot 1: Loss
    if not df_train.empty and 'epoch' in df_train.columns and 'loss' in df_train.columns:
        sns.lineplot(data=df_train, x='epoch', y='loss', ax=axes[0], color='red', label='Train Loss')
    if not df_eval.empty and 'epoch' in df_eval.columns and 'eval_loss' in df_eval.columns:
        sns.lineplot(data=df_eval, x='epoch', y='eval_loss', ax=axes[0], color='orange', label='Eval Loss', marker='o')
        
    axes[0].set_title(f'Loss over Epochs')
    axes[0].set_ylabel('Loss')
    if axes[0].get_legend(): axes[0].legend()

    # Plot 2: Learning Rate
    if not df_train.empty and 'epoch' in df_train.columns and 'learning_rate' in df_train.columns:
        sns.lineplot(data=df_train, x='epoch', y='learning_rate', ax=axes[1], color='blue')
        
    axes[1].set_title('Learning Rate over Epochs')
    axes[1].set_ylabel('Learning Rate')

    # Plot 3: Token Accuracy
    if not df_train.empty and 'epoch' in df_train.columns and 'mean_token_accuracy' in df_train.columns:
        sns.lineplot(data=df_train, x='epoch', y='mean_token_accuracy', ax=axes[2], color='green', label='Train Accuracy')
    if not df_eval.empty and 'epoch' in df_eval.columns and 'eval_mean_token_accuracy' in df_eval.columns:
        sns.lineplot(data=df_eval, x='epoch', y='eval_mean_token_accuracy', ax=axes[2], color='lime', label='Eval Accuracy', marker='o')
        
    axes[2].set_title('Mean Token Accuracy over Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_xlabel('Epoch' if num_subplots == 3 else '')
    if axes[2].get_legend(): axes[2].legend()

    # Plot 4: ROUGE (If present)
    if has_rouge:
        for col, color in zip(['rouge1', 'rouge2', 'rougeL'], ['purple', 'magenta', 'brown']):
            if col in df_rouge_avg.columns:
                sns.lineplot(data=df_rouge_avg, x='epoch', y=col, ax=axes[3], label=col, marker='o', color=color)
        axes[3].set_title('Average ROUGE Scores over Epochs')
        axes[3].set_ylabel('Score')
        axes[3].set_xlabel('Epoch')
        axes[3].legend()

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot successfully generated and saved as {output_filename}")


# Execute the functions individually to avoid combining curves
plot_training_and_eval_logs("output_vlmft_initial.log", "initial_training_curves.png")
plot_training_and_eval_logs("output_vlmft_final.log", "final_training_curves.png")
plot_training_and_eval_logs("graphs\output_vlmft_vitunfrozen.log", "vit_unfrozen_exp.png")
plot_training_and_eval_logs("graphs\slurm_vlm-exp_15783969.out", "llm_head_only_exp.png")