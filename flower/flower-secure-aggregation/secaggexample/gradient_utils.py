import torch
import torch.nn.functional as F

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def compute_gradient(model, data, label, criterion):
    output = model(data)
    loss = criterion(output, label)
    gradients = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in gradients]

def apply_defense(gradients, strategy):
    if strategy == 'none':
        return gradients

    elif strategy == 'pruning':
        pruned = []
        for grad in gradients:
            num_prune = int(0.2 * grad.numel() + 0.5)
            original_shape = grad.shape
            flattened = torch.abs(grad.flatten())
            _, indices = torch.topk(flattened, num_prune, largest=False)
            flattened[indices] = 0
            pruned_grad = flattened.view(original_shape)
            pruned.append(pruned_grad)
        return pruned

    elif strategy == 'quantization':
        return [torch.round(g * 15) / 15 for g in gradients]  # 4-bit quantization

    elif strategy == 'noise':
        return [g + torch.normal(mean=0, std=0.001, size=g.shape, device=g.device) for g in gradients]

    else:
        raise ValueError(f"Unknown defense strategy: {strategy}")
