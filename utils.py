import torch
import os
import imp

def cons_full_grads(params, batch_size):
    """COnstruct the compact form of gradients caculated by all samples, which is used for projection.

    Parameters
    ----------
    params : torch.nn.Parameter
        Contain the gradients to form the compact form.
    batch_size : _type_
        The number of samples.

    Returns
    -------
    Tensor[batch_size, -1]
        The tensor of the compact form of the gradients.
    """
    n_grads = []
    for param in params:
        grad = param.grad1.view(batch_size, -1)
        n_grads.append(grad)
    return torch.cat(n_grads, dim=1)


def make_env(scenario_name,  benchmark=False):
    """Prepare the interacted environment.

    Parameters
    ----------
    scenario_name : str
        The scenario where agents are placed.
    benchmark : bool, optional
        Whether using benchmark data or not, by default False

    Returns
    -------
    MultiAgentEnv
        The envrionment for interacting.
    """
    from multiagent.environment_no_diff_dist_partial import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    pathname = os.path.join(os.path.join(os.path.dirname(__file__), 'scenarios'), scenario_name + '.py')
    scenario = imp.load_source('', pathname).Scenario()
    
    # create world
    world = scenario.make_world()

    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)

    return env

def copy_flat_gradients(params):
    """Copy flattened gradients.

    Parameters
    ----------
    params : torch.nn.Parameter
        Parameters containing gradients. 

    Returns
    -------
    Tensor[-1]
        The tensor of falttened gradient.
    """
    new_grads = []

    for param in params:
        g = torch.clone(param.grad).detach().view(-1)
        new_grads.append(g)
    new_grads = torch.cat(new_grads)
    return new_grads


def reverse_flat_grad(grad_dict, flat_grad):
    """Reverse the flatten gradient to the corresponding parameters.

    Parameters
    ----------
    grad_dict : dict
        The dictionary containing the name and value of gradients.
    flat_grad : Tensor[-1]
        The falttened gradient.

    Returns
    -------
    dict
        The dictonary of reversed gradient.
    """
    temp_flat_size = []
    for grad_name in grad_dict:
        para_size = grad_dict[grad_name].nelement()
        temp_flat_size.append(para_size)
    start = 0
    index = 0
    new_grads = {}
    for grad_name in grad_dict:
        offset = temp_flat_size[index]
        grad  = torch.reshape(flat_grad[start:start+offset], shape=grad_dict[grad_name].shape)
        new_grads[grad_name] = grad
        start += offset
        index += 1
    return new_grads

def project(A, b):
    """Project b back to the subspace desribed by A to resist Orthogonal attack.

    Parameters
    ----------
    A : Tensor[batch_size, dim]
        The gradients to form the subspace.
    b : Tensor[dim]
        The vector to be projected.

    Returns
    -------
    Tensor[dim]
        Projected vector.
    """
    q,r = torch.linalg.qr(A.T)
    _b = torch.matmul(torch.matmul(q,q.T), b)
    return _b

def geometric_median(wList):
    """Calculate the gemoetric median of a list of vectors.

    Parameters
    ----------
    wList : List[Tensor]
        The list of caculated vectors.

    Returns
    -------
    Tensor
        The approximated geometric median.
    """
    max_iter = 80
    tol = 1e-5
    guess = torch.mean(wList, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack([w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
    return guess

def geometry_median_aggregation(grad_dict, num_agent):
    """Apply GeoMed to aggregate gradients.

    Parameters
    ----------
    grad_dict : dict
        The dictionary containing the name and the value of gradients.
    num_agent : int
        The number of agents.

    Returns
    -------
    Tensor[-1]
        The flatted gradient after aggregation.
    """
    temp_flat_grads = [[] for _ in range(num_agent)]
    temp_flat_size = [[] for _ in range(num_agent)]
    for gradient_name in grad_dict:
        for i, gradient in enumerate(grad_dict[gradient_name]):
            para_size = gradient.nelement()
            temp_flat_grads[i].append(gradient.view(-1))
            temp_flat_size[i].append(para_size)
    
    torch_flat_grads = [torch.cat(x) for x in temp_flat_grads]
    torch_flat_grads = torch.stack(torch_flat_grads, dim=0)
    g_median = geometric_median(torch_flat_grads)
    new_flat_grads = {}
    start = 0
    pos = 0 
    for para_name in grad_dict:
        flat_grad = g_median[start:start+temp_flat_size[i][pos]]
        para_grad = torch.reshape(flat_grad, shape=grad_dict[para_name][0].shape)
        new_flat_grads[para_name] = para_grad
        start += temp_flat_size[i][pos]
        pos += 1

    return new_flat_grads

def aggregate_grads(trainers, device, is_project=False):
    """Apply the average aggregation rule.

    Parameters
    ----------
    trainers : torch.nn.Module
        Agents for training
    device : str
        The device where models are placed
    is_project : bool, optional
        Whether using projection or not, by default False

    Returns
    -------
    _type_
        _description_
    """
    q_network_state_dict = trainers[0].q_network.state_dict()

    temp_grads = {}
    for para_name in q_network_state_dict:
        para = q_network_state_dict[para_name]
        temp_grads[para_name] = torch.zeros(size=para.shape, device=device, dtype=torch.float)
    
    for i, agent in enumerate(trainers):
        gradient_dict = {k:v.grad for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        for para_name in gradient_dict:
            temp_grads[para_name] += gradient_dict[para_name]    
    for para_name in temp_grads:
        temp_grads[para_name] /= len(trainers)
    
    flat_grad = []
    for grad_name in temp_grads:
        flat_grad.append(temp_grads[grad_name].view(-1))
    flat_grad = torch.cat(flat_grad)
    
    
    
    for i, agent in enumerate(trainers):
        if is_project:
            project_grad = project(agent.full_grads, flat_grad)
            new_grads = reverse_flat_grad(temp_grads, project_grad)
        else:
            new_grads = temp_grads
        gradient_dict = {k:v.grad for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        for para_name in gradient_dict:
            gradient_dict[para_name].copy_(new_grads[para_name])
    return torch.linalg.matrix_rank(agent.full_grads)