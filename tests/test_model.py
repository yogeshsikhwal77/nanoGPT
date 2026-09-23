import torch 
from nanogpt.config import ModelConfig
from nanogpt.model import GPT

cfg = ModelConfig(dim=64,layers=2,heads=4,context_len=32,vocab_size=100)

def test_shapes():
    m = GPT(cfg)
    idx = torch.randint(0,100,(3,16))
    logits,loss = m(idx,idx)
    assert logits.shape == (3,16,100)
    assert loss.ndim == 0


def test_weight_tying():
    m = GPT(cfg)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()

def test_causality():
    m = GPT(cfg).eval()
    a = torch.randint(0, 100, (1, 16))
    b = a.clone()
    b[0, 10:] = torch.randint(0, 100, (6,))   # change only the future
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)

def test_ignore_index():
    m = GPT(cfg)
    idx = torch.randint(0, 100, (2, 8))
    targets = idx.clone()
    targets[:, :4] = -100
    _, loss = m(idx, targets)
    assert torch.isfinite(loss)