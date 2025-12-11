from dpl.core import Variable
from dpl.functions.attention_weight import attention_weight
from dpl.functions.weight_sum import weight_sum


def attention(hs: Variable, h: Variable) -> tuple[Variable, Variable]:
    """Compute attention context vector.

    Args:
        hs: Encoder hidden states (batch_size, seq_len, hidden_size)
        h: Decoder hidden state (batch_size, hidden_size)

    Returns:
        c: Context vector (batch_size, hidden_size)
        a: Attention weights (batch_size, seq_len)
    """
    a = attention_weight(hs, h)
    c = weight_sum(hs, a)
    return c, a
