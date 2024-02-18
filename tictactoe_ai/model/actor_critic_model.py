from jax import numpy as jnp
from flax import linen as nn
from .mlp import mish


class ActorCriticModel(nn.Module):
    body: nn.Module
    actor_head: nn.Module
    critic_head: nn.Module
    
    @nn.compact
    def __call__(self, inputs, mask):
        x = inputs
        x = self.body(x)
        x = mish(x)
        
        actor_logits = self.actor_head(x)
        critic_value = self.critic_head(x)
        
        actor_logits = jnp.where(mask, actor_logits, -jnp.inf)
        return actor_logits, jnp.squeeze(critic_value)
