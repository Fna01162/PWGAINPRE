import torch

def count_parameters(model):
    """统计模型参数量，返回总参数量和可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_stats(G, D, R=None):
    """打印生成器/判别器（可选 Refiner）的参数量统计"""
    g_total, g_train = count_parameters(G)
    d_total, d_train = count_parameters(D)
    print(f"生成器参数量: {g_total/1e6:.2f}M (可训练: {g_train/1e6:.2f}M)")
    print(f"判别器参数量: {d_total/1e6:.2f}M (可训练: {d_train/1e6:.2f}M)")
    if R is not None:
        r_total, r_train = count_parameters(R)
        print(f"Residual Refiner 参数量: {r_total/1e6:.2f}M (可训练: {r_train/1e6:.2f}M)")
        print(f"总参数量: {(g_total+d_total+r_total)/1e6:.2f}M")
    else:
        print(f"总参数量: {(g_total+d_total)/1e6:.2f}M")