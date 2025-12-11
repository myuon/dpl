from dpl.core import Variable
import dpl.functions as F


def attention_weight(hs: Variable, h: Variable) -> Variable:
    """Compute attention weights using dot-product attention.

    Args:
        hs: Encoder hidden states (batch_size, seq_len, hidden_size)
        h: Decoder hidden state (batch_size, hidden_size)

    Returns:
        Attention weights (batch_size, seq_len)
    """
    batch_size, seq_len, hidden_size = hs.shape

    # Expand h for broadcasting: (batch_size, 1, hidden_size)
    hr = F.reshape(h, (batch_size, 1, hidden_size))

    # Element-wise product: (batch_size, seq_len, hidden_size)
    t = hs * hr

    # Sum over hidden dimension: (batch_size, seq_len)
    s = F.sum(t, axis=2)

    # Softmax over sequence dimension: (batch_size, seq_len)
    a = F.softmax(s, axis=1)

    return a
