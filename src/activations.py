import flax.linen as nn

class ParametricGatedActivation(nn.Module):
    """
    A custom gated activation function with learnable parameters.

    The activation function is defined as:
    a(x) = [γ + sigmoid(β * x) * (1 - γ)] * x
    where γ and β are learnable parameters.
    """
    features: int

    @nn.compact
    def __call__(self, x):
        """
        Applies the parametric gated activation function.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying the activation function.
        """
        beta = self.param('beta', nn.initializers.ones, (self.features,))
        gamma = self.param('gamma', nn.initializers.zeros, (self.features,))

        sigmoid_term = nn.sigmoid(beta * x)
        gate = gamma + sigmoid_term * (1 - gamma)
        
        return gate * x 