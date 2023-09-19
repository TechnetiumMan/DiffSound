import torch # the custom gradient layer of torch.unique
class CustomUnique(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        unique_values, indices, counts = torch.unique(input, dim=0, return_inverse=True, return_counts=True)
        ctx.save_for_backward(input, unique_values, counts)   
        return unique_values, indices

    @staticmethod
    def backward(ctx, grad_output, grad_output_indices):
        input, unique_values, counts = ctx.saved_tensors
        grad_input = torch.zeros_like(input, dtype=torch.float) # (ni, 3)
        # grad_input.scatter_(0, indices, grad_output)
        
        idx = (input == unique_values[:,:, None]).nonzero() # BUG!!!
        input_idx = idx[:, 1:] # (nidx, 2)
        output_idx = idx[:, 0] # (nidx,)
        weighted_grad = grad_output / counts[:,None] # (no, 3)
        
        # grad_input[idx[:,1:]] = weighted_grad[idx[:,0]].float()
        grad_input_flatten = grad_input.reshape(-1) # (ni*3,)
        idxed_weighted_grad = weighted_grad[output_idx].reshape(-1) # (nidx*3,)
        # input_idx_flatten = 
        raise NotImplementedError
        return grad_input
    
custom_unique = CustomUnique.apply