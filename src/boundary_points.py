import torch

def get_boundary_points(x_ini, x_fin, y_ini, y_fin, n_points_x, n_points_y, device = "cpu"):
    x_left = torch.full((n_points_y,), x_ini).reshape(-1, 1).requires_grad_(True).to(device)
    y_left = torch.linspace(y_ini, y_fin, n_points_y).reshape(-1, 1).requires_grad_(True).to(device)
    x_right = torch.full((n_points_y,), x_fin)
    y_right = torch.linspace(y_ini, y_fin, n_points_y)
    x_down = torch.linspace(x_ini, x_fin, n_points_x)
    y_down = torch.full((n_points_x,), y_ini)
    x_up = torch.linspace(x_ini, x_fin, n_points_x)
    y_up = torch.full((n_points_x,), y_fin)

    x_around = torch.cat((x_right, x_up, x_down)).reshape(-1, 1).requires_grad_(True).to(device)
    y_around = torch.cat((y_right, y_up, y_down)).reshape(-1, 1).requires_grad_(True).to(device)

    return x_around, y_around, x_left, y_left