import torch

# MLS Score
def mls(logits):
    scores = - torch.max(logits, dim=1)[0]
    return scores.cpu().numpy()

# MSP Score
def msp(logits):
    probas = torch.softmax(logits, dim=1)
    max_probas_scores = - torch.max(probas, dim=1)[0]
    return max_probas_scores.cpu().numpy()

# Energy Score
def energy(logits, temp=1):
    energies = - temp * torch.logsumexp(logits / temp, dim=1)
    return energies.cpu().numpy()

# Entropy Score
def entropy(logits):
    probas = torch.softmax(logits, dim=1)
    entropies = - torch.sum(probas * torch.log(probas + 1e-8), dim=1)
    return entropies.cpu().numpy()
