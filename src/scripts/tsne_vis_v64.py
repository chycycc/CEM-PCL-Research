import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.utils import config
from src.utils.data.loader import prepare_data_seq
from src.models.CEM.model import CEM

# 设置全局配置
config.model = 'cem'
config.test = True
config.model_path = 'save/test/CEM_19999_41.1517'  # 使用 v6.4 PPL 最优权重

def main():
    print("Loading data...")
    # 加载词表和数据集
    train_set, dev_set, test_set, vocab, num_classes = prepare_data_seq(
        batch_size=config.batch_size
    )

    print("Loading model...")
    # 初始化模型并加载权重
    model = CEM(
        vocab,
        decoder_number=32,  # 32 种情感
        is_eval=True,
        model_file_path=config.model_path
    )
    model.to(config.device)
    model.eval()

    emo_reps = []
    emo_labels = []

    print("Extracting emotion representations from test set...")
    with torch.no_grad():
        for batch in tqdm(test_set):
            # 前向传播，传入 need_rep=True 获取 emo_rep
            _, _, _, emo_rep = model.forward(batch, need_rep=True)
            labels = batch["program_label"]
            
            emo_reps.append(emo_rep.cpu().numpy())
            emo_labels.extend(labels)

    # 拼接所有批次的数据
    emo_reps = np.concatenate(emo_reps, axis=0)
    emo_labels = np.array(emo_labels)

    print(f"Extracted {emo_reps.shape[0]} samples.")

    # 提取模型中学习到的 32 个情感原型 (Prototypes)
    # epcl_criterion 是 PrototypeContrastiveLoss，里面的 prototypes 是 nn.Parameter
    prototypes = model.epcl_criterion.prototypes.detach().cpu().numpy()
    
    # ==== [修复: 必须进行 L2 标准化] ====
    # 在计算 InfoNCE/对比学习时，我们在球面上看角度。
    # 所以必须把它们除以自身的模长，消除欧氏距离导致的“假性坍塌”。
    emo_reps_norm = emo_reps / (np.linalg.norm(emo_reps, axis=1, keepdims=True) + 1e-8)
    prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    
    # 将标准化后的样本和原型拼接，保证降维在同一个空间进行
    all_reps = np.concatenate([emo_reps_norm, prototypes_norm], axis=0)
    
    print("Running t-SNE... (this may take a minute)")
    # 使用 t-SNE 降维至 2 维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all_reps_2d = tsne.fit_transform(all_reps)

    # 还原降维后的样本坐标和原型坐标
    emo_reps_2d = all_reps_2d[:len(emo_reps)]
    prototypes_2d = all_reps_2d[len(emo_reps):]

    print("Plotting...")
    plt.figure(figsize=(14, 10))
    
    # 绘制样本散点图
    scatter = plt.scatter(
        emo_reps_2d[:, 0], 
        emo_reps_2d[:, 1], 
        c=emo_labels, 
        cmap='tab20', 
        alpha=0.6, 
        s=15
    )
    
    # 绘制 32 个情感原型（用红色五角星表示）
    for i in range(32):
        plt.scatter(
            prototypes_2d[i, 0], 
            prototypes_2d[i, 1], 
            marker='*', 
            color='red', 
            s=300, 
            edgecolors='black',
            zorder=3 # 保证在最上层
        )
        # 标注原型的数字标签
        plt.text(
            prototypes_2d[i, 0] + 0.5, 
            prototypes_2d[i, 1] + 0.5, 
            str(i), 
            fontsize=12, 
            weight='bold',
            zorder=4
        )

    plt.colorbar(scatter, label='Emotion Category Index')
    plt.title('t-SNE Visualization of EPCL Emotion Manifold (v6.4)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 创建保存图片的文件夹
    output_dir = 'docs/figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'Fig_EPCL_tsne_v64.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved successfully to {save_path}!")

if __name__ == "__main__":
    main()
