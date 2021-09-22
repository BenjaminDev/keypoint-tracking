import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx]),
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class AWing(nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = (
            self.omega
            * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y)))
            * (self.alpha - y)
            * ((self.theta / self.epsilon) ** (self.alpha - y - 1))
            / self.epsilon
        )
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - y)
        )
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1
            + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon)
            ** (self.alpha - y[case1_ind])
        )
        lossMat[case2_ind] = (
            A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]
        )
        return lossMat


class Loss_weighted(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)

    def forward(self, y_pred, y, M):
        M = M.float()
        Loss = self.Awing(y_pred, y)
        weighted = Loss * (self.W * M + 1.0)
        return weighted.mean()


# Deprecated.
def wing_loss(
    output: torch.Tensor, target: torch.Tensor, width=5, curvature=0.5, reduction="mean"
):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param output:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - output).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


if __name__ == "__main__":

    out = torch.randn(4, 69, 64, 64)
    target = torch.randn(4, 69, 64, 64)
    M = torch.randn(4, 69, 64, 64)
    criterion = Loss_weighted()
    lossV = criterion(out, target, M)
    print(lossV)
    # loss vis
    # """
    import matplotlib.pyplot as plt

    lossmap = lossV.detach()[0, -1].numpy()
    plt.imshow(lossmap)
    plt.show()
    # """
