import os
from cGAN.data.datasets import CreateData, GanDataset


ABS_PATH = os.path.dirname(os.path.abspath(__file__))
def main(file_path):
    data = CreateData(file_path)
    datasets = GanDataset(data.x_data, data.y_data)
    

    


if __name__ == "__main__":
    main(file_path=f"{ABS_PATH}/datasets/Dataset_I.csv")
