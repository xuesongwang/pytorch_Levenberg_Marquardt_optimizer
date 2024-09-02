# torch_Levenberg_Marquardt_optimizer
Implementation of `trainlm` in Matlab that uses the Levenberg_Marquardt backpropagation for training neural networks. 

It has the efficiency advantage over stochastic gradient descents but is restricted to smaller networks. The repository is built on [torchimize](https://github.com/hahnec/torchimize) which enables convex optimization on GPUs based on the torch.Tensor class. Make sure to pip install torchimize before running this code.

Our contribution is to write a test code with the paramters being inside torch.nn.Module, which is the conventional way of defining neural networks in pytorch. The pipeline follows this [thread](https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240) to compute the Jocobian and update network parameters.

One can define their own `opt_loss_function` and `opt_jacobian_function` similar to ours and call the optimizer function `lsq_lma` to get the learned parameters:
```
p = torch.cat([param.view(-1).clone() for param in net.parameters()])

result = lsq_lma(
    p=p, 
    function=opt_loss_function,
    jac_function=opt_jacobian_function,
    args=(x, y, net),
    max_iter=500,
    gtol=1e-11
)

print("learned parameters",result[-1])
