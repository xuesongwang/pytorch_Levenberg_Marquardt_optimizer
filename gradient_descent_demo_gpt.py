from typing import List, Tuple
import torch
import torch.nn as nn
import copy
import torch.autograd.functional as F
from torchimize.functions import  lsq_lma

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = torch.cat([p.view(-1).detach().requires_grad_() for p in orig_params])
    return params, names, orig_params


def load_weights(mod: nn.Module, names: List[str], params, orig_params) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    start = 0

    for name, p in zip(names, orig_params):
        end = start + p.numel() # just need the index from orig_params, the data actually comes from params
        _set_nested_attr(mod, name.split("."), params[start:end].reshape(p.shape))
        start = end




class Net(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        # self.nn = nn.Sequential(nn.Linear(in_features=4, out_features=7, bias=True),
        #                         nn.Tanh(),
        #                         nn.Linear(in_features=7, out_features=3, bias=True))
        self.nn = nn.Linear(in_features=n_in, out_features=n_out, bias=False)

    def forward(self, x):
        return self.nn(x)


# Compute the loss
def compute_loss(y_pred, y):
    return torch.sum((y_pred - y) ** 2, dim=-1)


def jacobian_function(x, y, net):
    params, names, orig_params = extract_weights(net)
    def forward(*new_params: Tensor) -> Tensor:
        load_weights(net, names, new_params[0], orig_params)
        out = net(x)
        loss = compute_loss(out, y)
        return loss

    return forward, params


def opt_loss_function(p, x, y, net):
    # remove the paramters in net and get the tuple parameter through orig_params
    net_cp = copy.deepcopy(net)
    _, names, orig_params = extract_weights(net_cp)

    # reload net parameters with p
    load_weights(net_cp, names, p, orig_params)
    out = net_cp(x)
    loss = compute_loss(out, y)
    return loss

def opt_jacobian_function(p, x, y, net):
    net_cp = copy.deepcopy(net)
    def opt_jacobian_wrapper():
        _, names, orig_params = extract_weights(net_cp)
        def forward(*new_params: Tensor) -> Tensor:
            load_weights(net_cp, names, new_params[0], orig_params)
            out = net_cp(x)
            loss = compute_loss(out, y)
            return loss
        return forward, p

    j_func, p = opt_jacobian_wrapper()
    jacob_matrix = F.jacobian(j_func, p)
    return jacob_matrix

n_in, n_out = [4,3]
net = Net(n_in, n_out)
x = torch.randn(1000, n_in).requires_grad_()
p_ground_truth = torch.rand(n_in, n_out)
y = torch.matmul(x, p_ground_truth)


p = torch.cat([param.view(-1).clone() for param in net.parameters()])
result = lsq_lma(
    p=p,
    function=opt_loss_function,
    jac_function=opt_jacobian_function,
    args=(x, y, net),
    max_iter=500,
    gtol=1e-11
)

print("finished!")
print("how many updates", len(result))
print("ground_truth",p)
print("prediction",result[-1])
# only when your result[-1] equals the ground_truth
print("parameter error from the ground truth", torch.mean((result[-1].reshape(*p_ground_truth.shape)-p_ground_truth)**2))

loss = opt_loss_function(result[-1], x, y, net)
print("loss", loss.mean().item())