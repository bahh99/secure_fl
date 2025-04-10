# secaggexample/attack.py

def perform_inversion_attack(model, gradients, gt_label, criterion, num_iters=300):
    dummy_data = torch.randn((1, 3, 32, 32), requires_grad=True)
    dummy_label = torch.randn((1, 100), requires_grad=True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    
    for _ in range(num_iters):
        def closure():
            optimizer.zero_grad()
            output = model(dummy_data)
            dummy_onehot = F.softmax(dummy_label, dim=-1)
            loss = criterion(output, dummy_onehot)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grad_diff = sum(((dg - og) ** 2).sum() for dg, og in zip(dummy_grads, gradients))
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
    
    return dummy_data.detach(), dummy_label.detach()
