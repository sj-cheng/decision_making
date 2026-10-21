
# standard package
import torch
import numpy as np 
from torch.nn import MSELoss
import torch.nn.functional as F

# my package
from learning.feedforward import FeedForward

class GaussianPolicyNetwork(torch.nn.Module):

	def __init__(self,problem,robot,device="cpu",path=None):
		super(GaussianPolicyNetwork, self).__init__()

		h = 64

		self.encoding_dim = problem.policy_encoding_dim
		self.output_dim = 2*len(problem.action_idxs[robot]) 
		self.state_dim = problem.state_dim 
		self.action_dim = problem.action_dim 
		self.action_lims = problem.action_lims
		self.robot_action_idxs = problem.action_idxs[robot]
		self.robot = int(robot)
		# Episodes randomize speed limits, while the policy input does not carry
		# that limit.  Use a fixed reference range during fitting so targets from
		# the upper end of the episode range are reachable; eval() still rescales
		# the same raw policy to the current episode limits.
		self.reference_action_lims = np.array(self.action_lims, copy=True)
		if self.robot in getattr(problem, "evaders", []):
			max_speed = float(getattr(problem, "evader_speed_lim_range", (0.0, 1.0))[1])
		else:
			max_speed = float(getattr(problem, "pursuer_speed_lim_range", (0.0, 1.0))[1])
		self.reference_action_lims[self.robot_action_idxs, 0] = -max_speed
		self.reference_action_lims[self.robot_action_idxs, 1] = max_speed
		self.is_evader = self.robot in getattr(problem, "evaders", [])
		# Direction is more important than matching a small component-wise
		# average for evaders, where several valid escape headings otherwise
		# regress toward a short near-zero vector.
		self.direction_loss_weight = 0.50 if self.is_evader else 0.0
		# Component-wise MSE can still shrink an evader action when nearby
		# states have several valid escape headings.  Keep a separate radial
		# loss so the network cannot satisfy the direction term with a tiny
		# near-zero vector.
		self.magnitude_loss_weight = 1.50 if self.is_evader else 0.0
		self.direction_target_threshold = 0.10 * float(
			np.mean(self.action_lims[self.robot_action_idxs, 1] - self.action_lims[self.robot_action_idxs, 0])
		)
		self.device = torch.device(device)
		self.path = path
		self.name = "gaussian"

		psi_network_architecture = [
			["Linear", self.encoding_dim, h], 
			["Linear", h, h],
			["Linear", h, h],
			["Linear", h, self.output_dim] 
		]

		self.psi = FeedForward(
			psi_network_architecture,
			"relu",
			device)	

		self.to(self.device)

		if path is not None:
			print("Successful loading of GaussianPolicyNetwork from {}.".format(path))
			self.load_state_dict(torch.load(path))


	def to(self, device):
		self.device = device
		self.psi.to(device)
		return super().to(device)

	def sync_action_lims(self, problem):
		self.action_lims = problem.action_lims
		return self.action_lims


	def forward(self,x,training=False):
		dist = self.psi(x)
		mu, logvar = torch.split(dist, 2, dim=1)

		if training:
			return mu, logvar
		else:
			batch_size = x.shape[0]
			# logvar is trained in physical action units.  Sampling after the
			# tanh scaling keeps the variance interpretation consistent with the
			# visit-distribution targets and with the C++ wrapper.
			mean = self.scale_action(mu)
			logvar = torch.clamp(logvar, min=-8.0, max=2.0)
			sd = torch.sqrt(torch.exp(logvar))
			eps = torch.randn(size=(batch_size, int(self.output_dim / 2)), device=self.device)
			policy = mean + sd * eps
			u = torch.as_tensor(self.action_lims[self.robot_action_idxs, 1], dtype=policy.dtype, device=policy.device).unsqueeze(0)
			l = torch.as_tensor(self.action_lims[self.robot_action_idxs, 0], dtype=policy.dtype, device=policy.device).unsqueeze(0)
			return torch.maximum(torch.minimum(policy, u), l)
			
	def loss_fnc(self,x,target):
		# mu,logvar = self.__call__(x,training=True)
		# criterion = MSELoss(reduction='none')
		# loss = torch.sum(criterion(mu, target) / (2*torch.exp(logvar)) + 1/2*logvar)
		
		# https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf
		# likelihood = -N/2 log det(Var) - 1/2 sum_{i=0}^{N} (x_i - mu)^T Var^{-1} (x_i - mu)
		# 	- log( det(Var)) = log( Var(0,0) * Var(1,1) * ... ) = log(Var(0,0)) + log(Var(1,1)) + ... 
		# 		- for diagonal matrix, det(Var) = Var(0,0) * Var(1,1) * ... 
  
		mu,logvar = self(x,training=True)
		action_dim = int(self.output_dim / 2)
		if target.shape[1] == action_dim:
			# Backward-compatible loss for old two-value datasets.
			scaled_mu = self.scale_action(mu, action_lims=self.reference_action_lims)
			mean_loss = (scaled_mu - target).pow(2)
			target_mean = target
			pred_mean = scaled_mu
			pred_logvar = None
			target_logvar = None
		else:
			if target.shape[1] != 2 * action_dim:
				raise ValueError("policy target must contain action mean or mean/logvar")
			target_mean = target[:, :action_dim]
			target_logvar = target[:, action_dim:]
			pred_mean = self.scale_action(mu, action_lims=self.reference_action_lims)
			pred_logvar = torch.clamp(logvar, min=-8.0, max=2.0)
			mean_loss = (pred_mean - target_mean).pow(2)

		loss = torch.mean(mean_loss)
		if target_logvar is not None:
			variance_loss = (pred_logvar - target_logvar).pow(2)
			loss = loss + 0.25 * torch.mean(variance_loss)

		if self.direction_loss_weight > 0.0:
			target_norm = torch.linalg.vector_norm(target_mean, dim=1)
			mask = target_norm > self.direction_target_threshold
			if torch.any(mask):
				action_range = torch.as_tensor(
					self.reference_action_lims[self.robot_action_idxs, 1] - self.reference_action_lims[self.robot_action_idxs, 0],
					dtype=pred_mean.dtype,
					device=pred_mean.device,
				)
				norm_scale = torch.mean(action_range).clamp_min(1e-6)
				pred_norm = torch.linalg.vector_norm(pred_mean[mask], dim=1)
				target_norm_masked = target_norm[mask]
				magnitude_loss = ((pred_norm - target_norm_masked) / norm_scale).pow(2)
				loss = loss + self.magnitude_loss_weight * torch.mean(magnitude_loss)
				cosine_loss = 1.0 - F.cosine_similarity(
					pred_mean[mask], target_mean[mask], dim=1, eps=1e-6
				)
				loss = loss + self.direction_loss_weight * torch.mean(cosine_loss)
		return loss

	def eval(self,problem,root_state,robot,stochastic=True):
		super().eval()
		self.sync_action_lims(problem)
		policy_encoding = problem.policy_encoding(root_state,robot)
		policy_encoding = torch.tensor(policy_encoding,dtype=torch.float32).squeeze().unsqueeze(0) # [batch_size x state_dim]
		if stochastic:
			policy = self(policy_encoding)
		else:
			mu, _ = self(policy_encoding, training=True)
			policy = self.scale_action(mu)
		policy = policy.detach().numpy().reshape(int(self.output_dim/2),1) # [action_dim_per_robot x 1]
		super().train()
		return policy 

	def scale_action(self,action,action_lims=None):
		if action_lims is None:
			action_lims = self.action_lims
		u = torch.as_tensor(
			action_lims[self.robot_action_idxs,1],
			dtype=action.dtype,
			device=action.device,
		).unsqueeze(axis=0)
		l = torch.as_tensor(
			action_lims[self.robot_action_idxs,0],
			dtype=action.dtype,
			device=action.device,
		).unsqueeze(axis=0)
		center = 0.5 * (u + l)
		half_range = 0.5 * (u - l)
		return center + half_range * torch.tanh(action)
