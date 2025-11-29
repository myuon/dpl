from dpl.core import Variable, Function, ndarray, get_array_module
import dpl.functions as F


class NegativeSamplingLoss(Function):
    """
    Negative Sampling Loss for efficient word embedding training.

    This loss function is more efficient than softmax cross entropy for large vocabularies.
    Instead of computing probabilities over all words, it only considers:
    - The target word (positive sample)
    - A small number of negative samples

    Loss = -log(sigmoid(target_score)) - sum(log(sigmoid(-negative_scores)))
    """

    def __init__(self, sample_size: int = 5):
        """
        Args:
            sample_size: Number of negative samples to use
        """
        self.sample_size = sample_size
        self.negative_labels = None

    def forward(self, *xs: ndarray) -> ndarray:
        """
        Args:
            xs[0]: Target scores. Shape: (batch_size, 1)
            xs[1]: Negative scores. Shape: (batch_size, sample_size)

        Returns:
            Loss value (scalar)
        """
        target_score, negative_scores = xs
        batch_size = target_score.shape[0]
        xp = get_array_module(target_score)

        # Compute loss:
        # Loss = -log(sigmoid(target_score)) - sum(log(sigmoid(-negative_scores)))
        # Using log-sigmoid for numerical stability: log(sigmoid(x)) = -log(1 + exp(-x))

        # Positive loss: -log(sigmoid(target_score))
        positive_loss = xp.sum(-xp.log(1 / (1 + xp.exp(-target_score)) + 1e-7))

        # Negative loss: -sum(log(sigmoid(-negative_scores)))
        negative_loss = xp.sum(-xp.log(1 / (1 + xp.exp(negative_scores)) + 1e-7))

        loss = (positive_loss + negative_loss) / batch_size

        return xp.array(loss)

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        """
        Compute gradients with respect to target_score and negative_scores.

        d/dx log(sigmoid(x)) = 1 - sigmoid(x) = -1 + sigmoid(x)
        d/dx log(sigmoid(-x)) = -sigmoid(x)

        For our loss:
        d/d(target_score) [-log(sigmoid(target_score))] = -(1 - sigmoid(target_score)) = sigmoid(target_score) - 1
        d/d(negative_scores) [-log(sigmoid(-negative_scores))] = sigmoid(negative_scores)
        """
        (gy,) = gys
        target_score, negative_scores = self.inputs
        batch_size = target_score.shape[0]

        # Gradient for target: sigmoid(target_score) - 1
        target_grad = F.sigmoid(target_score) - 1.0

        # Gradient for negatives: sigmoid(negative_scores)
        negative_grad = F.sigmoid(negative_scores)

        # Scale by batch size and gy
        target_grad = target_grad * (gy / batch_size)
        negative_grad = negative_grad * (gy / batch_size)

        return target_grad, negative_grad


def negative_sampling_loss(
    target_score: Variable | ndarray,
    negative_scores: Variable | ndarray,
    sample_size: int = 5
) -> Variable:
    """
    Negative Sampling Loss function.

    Args:
        target_score: Scores for target words. Shape: (batch_size, 1)
        negative_scores: Scores for negative samples. Shape: (batch_size, sample_size)
        sample_size: Number of negative samples

    Returns:
        Loss value
    """
    return NegativeSamplingLoss(sample_size=sample_size)(target_score, negative_scores)
