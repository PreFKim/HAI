import torch

class EMA:
    def __init__(self, model, ema_model, max_alpha):
        self.ema_model = ema_model
        self.model = model
        print(ema_model.load_state_dict(model.state_dict()))
        for param in ema_model.parameters():
            param.requires_grad_(False)
        self.max_alpha = max_alpha
        self.step = 1
        self.alpha = 0.

    def update_alpha(self):
        self.alpha = min(1 - 1 / (self.step + 1), self.max_alpha)
    
    def update_weight(self):
        self.update_alpha()
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.mul_(self.alpha).add_(model_param, alpha=1.0 - self.alpha)
                    
        self.step += 1

    def state_dict(self):
        return self.ema_model.state_dict()