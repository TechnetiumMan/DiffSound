import torch

# Sample input tensor x
x = torch.tensor([[0, 0],[0, 0],[0, 1]])

# Count the number of repetitions for each element in x
unique_values, counts = torch.unique(x, return_counts=True)

# Create a tensor with the same shape as x to hold the result
y = torch.zeros_like(x, dtype=torch.float)

# Use broadcasting to calculate the values in y
idx = (x == unique_values[:, None]).nonzero()
print(idx)
y[idx[:,1]] = 1.0 / counts[idx[:,0]].float()

# Print the result
print(y)

# the custom gradient layer of torch.unique
class custom_unique(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        unique_values, indices, counts = torch.unique(input, sorted=True, return_inverse=True, return_counts=True)
        ctx.save_for_backward(input, unique_values, counts)   
        return unique_values, indices

    @staticmethod
    def backward(ctx, grad_output, grad_output_indices):
        input, unique_values, counts = ctx.saved_tensors
        grad_input = torch.zeros_like(input, dtype=torch.float)
        # grad_input.scatter_(0, indices, grad_output)
        
        idx = (input == unique_values[:, None]).nonzero()
        grad_input[idx[:,1]] = (grad_output / counts)[idx[:,0]].float()
        
        return grad_input