from dataset import prepare_data
from params import params
from train import train
from model import MarketPredictionModel

train_dataset, validation_dataset = prepare_data()

model = MarketPredictionModel()

train(model, (train_dataset, validation_dataset))
