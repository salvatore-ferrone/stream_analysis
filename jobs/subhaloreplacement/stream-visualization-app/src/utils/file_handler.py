def load_plot(plot_name):
    import os
    import matplotlib.pyplot as plt

    plot_path = os.path.join("data", "plots", f"{plot_name}.png")
    
    if os.path.exists(plot_path):
        img = plt.imread(plot_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        raise FileNotFoundError(f"Plot file {plot_name}.png not found in {plot_path}.")

def check_file_exists(file_path):
    return os.path.exists(file_path)