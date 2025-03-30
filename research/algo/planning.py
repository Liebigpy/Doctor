import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


def pytorch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


class cem_planner():
    """
    Cross Entropy Method control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    def __init__(self, model, nx, nu, running_cost = None,
                 num_samples=100, num_elite=2, horizon=4,
                 device="cuda",
                 tokenizer_manager = None,
                 terminal_state_cost=False,
                 u_min=None,
                 u_max=None,
                 choose_best=False,
                 init_cov_diag=1):

        self.model = model
        self.device = device
        self.tokenizer_manager = tokenizer_manager
        self.dtype = torch.float64
        self.N = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.num_elite = num_elite
        #self.choose_best = choose_best
        self.terminal_state_cost = terminal_state_cost

        # dimensions of state and control
        self.nx = nx
        self.nu = nu

        self.mean = None
        self.cov = None

        self.F = model
        self.running_cost = running_cost

        self.init_cov_diag = init_cov_diag
        self.u_min = u_min
        self.u_max = u_max
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.device)
            self.u_max = self.u_max.to(device=self.device)
        self.action_distribution = None

        # regularize covariance
        self.cov_reg = torch.eye(self.T * self.nu, device=self.device, dtype=self.dtype) * init_cov_diag * 1e-5

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        # action distribution, initialized as N(0,I)
        # we do Hp x 1 instead of H x p because covariance will be Hp x Hp matrix instead of some higher dim tensor
        self.mean = torch.zeros(self.T * self.nu, device=self.device, dtype=self.dtype)
        self.cov = torch.eye(self.T * self.nu, device=self.device, dtype=self.dtype) * self.init_cov_diag

    def _bound_samples(self, samples):
        if self.u_max is not None:
            for t in range(self.T):
                u = samples[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                samples[:, self._slice_control(t)] = cu
        return samples

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def _evaluate_trajectories_T(self, samples, torch_trajectories,torch_masks):
        # cost_total = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        # state = init_state.view(1, -1).repeat(self.K, 1)
        for t in range(self.T-1):
            u = samples[:, self._slice_control(t)]
            torch_trajectories["actions"][:, t, :] = u
            torch_masks['actions'][t] = 1
            # torch_trajectories["actions"] = decode["actions"].repeat(N, 1, 1)
            # add noise to the actions and clip
            torch_trajectories["actions"] += (
                    torch.randn_like(torch_trajectories["actions"], device=self.device) * 0.1
            )
            torch_trajectories["actions"] = torch.clamp(
                torch_trajectories["actions"], -1, 1
            )

            encoded_trajectories = self.tokenizer_manager.encode(torch_trajectories)
            predicted = self.model(encoded_trajectories, torch_masks)
            decode = self.tokenizer_manager.decode(predicted)
            torch_trajectories["states"][:, t+1, :] = decode["states"][:, t+1, :]
            torch_masks['states'][t+1] = 1
            value_return = decode["returns"].sum(dim=1)
            cost_total = value_return
            # state = self.F(state, u)
            # cost_total += self.running_cost(state, u)
        if self.terminal_state_cost:
            encoded_trajectories = self.tokenizer_manager.encode(torch_trajectories)
            predicted = self.model(encoded_trajectories, torch_masks)
            decode = self.tokenizer_manager.decode(predicted)
            value_return = decode["returns"].sum(dim=1)
            cost_total = value_return
            # cost_total += self.terminal_state_cost(state)
        return cost_total, decode

    def _evaluate_trajectories(self, samples, torch_trajectories,torch_masks,pos):
        # cost_total = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        # state = init_state.view(1, -1).repeat(self.K, 1)
        # print("debug170",self._slice_control(3))
        # u = samples[:, self._slice_control(3)]
        u = samples
        torch_trajectories["actions"][:, pos, :] = u
        torch_masks['actions'][pos] = 1
        torch_trajectories["actions"] = torch.clamp(
            torch_trajectories["actions"], -1, 1)
        encoded_trajectories = self.tokenizer_manager.encode(torch_trajectories)
        predicted = self.model(encoded_trajectories, torch_masks)
        decode = self.tokenizer_manager.decode(predicted)
        value_return = decode["returns"][:, pos].clone()

        return value_return, decode

    def _sample_top_trajectories(self,torch_trajectories, torch_masks, pos):
        # sample K action trajectories
        # in case it's singular
        self.action_distribution = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        samples = self.action_distribution.sample((self.N,))
        # bound to control maximums
        samples = self._bound_samples(samples)

        value_return, decode = self._evaluate_trajectories(samples, torch_trajectories, torch_masks,pos)

        #sorted_values, sorted_indices = torch.sort(value_return, descending=True)
        # select top k based on score
        #print("value_return",value_return,value_return.shape)
        value_return_flat = value_return.view(-1)
        sorted_values, topk = torch.topk(value_return_flat, k=self.num_elite, largest=True)
        new_decode = {k: v[topk[0]] for k,v in decode.items()}
        top_samples = samples[topk]
        return top_samples,new_decode


    def _sample_top_trajectories_explore(self,obs_tiled, critic,
                                         expl_noise: float = 0.03,
                                         noise_clip: float = 0.2):
        # sample K action trajectories
        # in case it's singular
        self.action_distribution = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        samples = self.action_distribution.sample((self.N,))
        # bound to control maximums
        # samples += (
        #         torch.randn_like(samples) * 0.1
        # )
        noise = (torch.randn_like(samples) * expl_noise).clamp(
            -noise_clip, noise_clip
        )
        samples += noise

        samples = self._bound_samples(samples)
        samples = torch.clamp(samples, -1, 1)


        with torch.no_grad():
            target_q = critic.q_target(obs_tiled, samples)
            v = critic.vf(obs_tiled)
        adv = target_q - v

        #value_return, decode = self._evaluate_trajectories(samples, torch_trajectories, torch_masks, pos)

        #sorted_values, sorted_indices = torch.sort(value_return, descending=True)
        # select top k based on score
        value_return_flat = adv.view(-1)
        sorted_values, topk = torch.topk(value_return_flat, k=self.num_elite, largest=True)
        #top_costs, topk = torch.topk(value_return, 1, largest=False, sorted=False)
        # new_decode = {k: v[topk[0]] for k,v in decode.items()}
        top_samples = samples[topk]
        return top_samples

    def _sample_top_trajectories_critic(self, torch_trajectories, torch_masks, pos,
                                        expl_noise: float = 0.03,
                                        noise_clip: float = 0.2):
        # sample K action trajectories
        # in case it's singular
        try:
            self.action_distribution = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        except ValueError as e:
            print("Covariance matrix is not positive definite:", e)
            epsilon = 1e-5
            self.cov = self.cov + epsilon * torch.eye(self.cov.size(-1), device=self.cov.device)

        samples = self.action_distribution.sample((self.N,))
        # bound to control maximums
        # samples += (
        #         torch.randn_like(samples) * 0.1
        # )
        noise = (torch.randn_like(samples) * expl_noise).clamp(
            -noise_clip, noise_clip
        )
        samples += noise

        samples = self._bound_samples(samples)
        samples = torch.clamp(samples, -1, 1)

        torch_trajectories["actions"][:, pos, :] = samples
        torch_trajectories["actions"] = torch.clamp(torch_trajectories["actions"], -1, 1)
        with torch.no_grad():
            # torch_trajectories["actions"] += (
            encoded_trajectories = self.tokenizer_manager.encode(torch_trajectories)
            _, value_outs = self.model(encoded_trajectories, torch_masks)

        [v_out, n_v_out, q_out, q_out_old] = value_outs
        adv = q_out_old - v_out
        # value_return_flat = adv[:, pos].view(-1)
        value_return_flat = q_out_old[:, pos].view(-1)

        # [v_out, n_v_out, q_out, q_out_old] = value_outs
        # value_return_flat = n_v_out[:, -1].view(-1)

        # [v_out, q_out, q_out_old] = value_outs
        # n_v_out = q_out - v_out
        # value_return_flat = n_v_out[:, pos].view(-1)

        # value_return, decode = self._evaluate_trajectories(samples, torch_trajectories, torch_masks, pos)

        # sorted_values, sorted_indices = torch.sort(value_return, descending=True)
        # select top k based on score

        sorted_values, topk = torch.topk(value_return_flat, k=self.num_elite, largest=True)
        # top_costs, topk = torch.topk(value_return, 1, largest=False, sorted=False)

        # new_decode = {k: v[topk[0]] for k,v in decode.items()}
        top_samples = samples[topk]
        return top_samples
