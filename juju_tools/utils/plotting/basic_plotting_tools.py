import matplotlib.pyplot as plt


def plot_xy(pair: list[tuple[float, float]]):
    x = [x[0] for x in pair]
    y = [y[1] for y in pair]
    plot_x_y(x, y)

def plot_x_y(x: list[float], y: list[float]):
    plt.plot(x, y, color='blue', label='Points')
    default_plot()

def default_plot(title: str = "X-Y Plot"):
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()