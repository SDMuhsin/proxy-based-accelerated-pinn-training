"""
Large-scale PINN models for testing LoRA speed hypothesis

Scales up the model to ~500K parameters to test if LoRA provides
speed improvements at larger scales (even though it will overfit badly).
"""

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable width"""
    def __init__(self, input_dim, output_dim, layers_num=3, hidden_dim=60, act=Sin()):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), act]
        for _ in range(layers_num - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Solution_u_Large(nn.Module):
    """
    Large solution network (~500K parameters)

    Architecture: 17 → 1024 → 1024 → 512 → 256 (encoder) → 128 → 64 → 1 (predictor)
    This gives us approximately:
    - encoder: 17*1024 + 1024*1024 + 1024*512 + 512*256 = ~1.7M params
    - Let's use: 17 → 512 → 512 → 256 → 128 for ~400K params
    """
    def __init__(self):
        super(Solution_u_Large, self).__init__()
        act = Sin()

        # Encoder: 17 → 512 → 512 → 256 → 128
        # Params: 17*512 + 512*512 + 512*256 + 256*128 = 8,704 + 262,144 + 131,072 + 32,768 = 434,688
        self.encoder = MLP(input_dim=17, output_dim=128, layers_num=5, hidden_dim=512, act=act)

        # Predictor: 128 → 64 → 32 → 1
        # Params: 128*64 + 64*32 + 32*1 = 8,192 + 2,048 + 32 = 10,272
        self.predictor = nn.Sequential(
            nn.Linear(128, 64), act,
            nn.Linear(64, 32), act,
            nn.Linear(32, 1)
        )

        # Total: ~445K parameters

    def forward(self, xt):
        encoded = self.encoder(xt)
        output = self.predictor(encoded)
        return output


class PINN_Large(nn.Module):
    """Large PINN with ~500K parameters in solution_u"""

    def __init__(self, args):
        super(PINN_Large, self).__init__()

        self.args = args
        self.solution_u = Solution_u_Large().to(device)

        # Keep dynamical_F small (frozen during fine-tuning anyway)
        self.dynamical_F = MLP(input_dim=35, output_dim=1,
                              layers_num=args.F_layers_num,
                              hidden_dim=args.F_hidden_dim,
                              act=Sin()).to(device)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # Print parameter counts
        solution_u_params = sum(p.numel() for p in self.solution_u.parameters())
        dynamical_F_params = sum(p.numel() for p in self.dynamical_F.parameters())
        print(f"\nLarge PINN Architecture:")
        print(f"  solution_u parameters: {solution_u_params:,}")
        print(f"  dynamical_F parameters: {dynamical_F_params:,}")
        print(f"  Total parameters: {solution_u_params + dynamical_F_params:,}")

    def predict(self, xt):
        return self.solution_u(xt)

    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def forward(self, xt):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]

        u = self.solution_u(torch.cat((x, t), dim=1))

        u_t = grad(u.sum(), t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(), x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        f = u_t - F
        return u, f

    def Train(self, trainloader, validloader, testloader):
        """Simplified training for POC"""
        optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=self.args.warmup_lr)
        optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=self.args.lr_F)

        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=self.args.epochs, eta_min=self.args.final_lr
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=self.args.epochs, eta_min=self.args.final_lr
        )

        for epoch in range(1, self.args.epochs + 1):
            self.train()
            for iter, (x1, x2, y1, y2) in enumerate(trainloader):
                x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
                u1, f1 = self.forward(x1)
                u2, f2 = self.forward(x2)

                loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
                f_target = torch.zeros_like(f1)
                loss2 = 0.5 * self.loss_func(f1, f_target) + 0.5 * self.loss_func(f2, f_target)
                loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                loss = loss1 + self.args.alpha * loss2 + self.args.beta * loss3

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()

            scheduler1.step()
            scheduler2.step()

            # Log every 5 epochs
            if epoch % 5 == 0 or epoch == self.args.epochs:
                mse = self.Valid(validloader)
                logger.info(f'[Train] epoch:{epoch}, valid MSE: {mse:.6f}')

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()
