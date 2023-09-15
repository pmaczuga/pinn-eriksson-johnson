import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

X_INI = 0.0
X_FIN = 1.0
Y_INI = 0.0
Y_FIN = 1.0

NUM_POINTS_X = 100
NUM_POINTS_Y = 100

PLOT_POINTS = 1000

EPSILON = 1000  # <-HERE W CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

NUMBER_EPOCHS = 1000