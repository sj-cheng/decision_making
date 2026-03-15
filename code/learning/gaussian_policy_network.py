
# standard package
import torch
import numpy as np 
from torch.nn import MSELoss

# my package
from learning.feedforward import FeedForward

class GaussianPolicyNetwork(torch.nn.Module):

	def __init__(self,problem,robot,device="cpu",path=None):
		super(GaussianPolicyNetwork, self).__init__()

		h = 24

		self.encoding_dim = problem.policy_encoding_dim
		self.output_dim = 2*len(problem.action_idxs[robot]) 
		self.state_dim = problem.state_dim 
		self.action_dim = problem.action_dim 
		self.action_lims = problem.action_lims
		self.robot_action_idxs = problem.action_idxs[robot]
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


	def forward(self,x,training=False):
		x[:,8] = 0 # zero out the last element of the encoding, which is the time encoding, to make the policy time-invariant.
		dist = self.psi(x)
		mu, logvar = torch.split(dist, 2, dim=1)

		if training:
			return mu, logvar
		else:
			batch_size = x.shape[0]
			# sd = torch.sqrt(torch.exp(logvar))
			eps = torch.randn(size=(batch_size,int(self.output_dim/2)),device=self.device)
			policy = mu + 0.1 * eps
			policy = self.scale_action(policy)
			return policy
			
	def loss_fnc(self,x,target):
		# mu,logvar = self.__call__(x,training=True)
		# criterion = MSELoss(reduction='none')
		# loss = torch.sum(criterion(mu, target) / (2*torch.exp(logvar)) + 1/2*logvar)
		
		# https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf
		# likelihood = -N/2 log det(Var) - 1/2 sum_{i=0}^{N} (x_i - mu)^T Var^{-1} (x_i - mu)
		# 	- log( det(Var)) = log( Var(0,0) * Var(1,1) * ... ) = log(Var(0,0)) + log(Var(1,1)) + ... 
		# 		- for diagonal matrix, det(Var) = Var(0,0) * Var(1,1) * ... 
  
		mu,logvar = self(x,training=True)
		# loss = torch.sum((mu - target).pow(2) / (2*torch.exp(logvar)) + mu.shape[0]/2 * logvar)
		loss = (mu - target).pow(2) # / (2*torch.exp(logvar)) + 1/2 * logvar)
		loss = torch.mean(loss)
		return loss 

	def eval(self,problem,root_state,robot):
		super().eval()
		self.action_lims = problem.action_lims
		policy_encoding = problem.policy_encoding(root_state,robot)
		policy_encoding = torch.tensor(policy_encoding,dtype=torch.float32).squeeze().unsqueeze(0) # [batch_size x state_dim]
		policy = self(policy_encoding).detach().numpy().reshape(int(self.output_dim/2),1) # [action_dim_per_robot x 1]
		super().train()
		return policy 

	def scale_action(self,action):
		u = torch.as_tensor(
			self.action_lims[self.robot_action_idxs,1],
			dtype=action.dtype,
			device=action.device,
		).unsqueeze(axis=0)
		l = torch.as_tensor(
			self.action_lims[self.robot_action_idxs,0],
			dtype=action.dtype,
			device=action.device,
		).unsqueeze(axis=0)
		center = 0.5 * (u + l)
		half_range = 0.5 * (u - l)
		return center + half_range * torch.tanh(action)