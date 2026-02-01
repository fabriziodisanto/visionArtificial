from utiles.hu_moments_tp import generate_hu_moments_file
from utiles.test_tp import load_and_test
from utiles.entrenando_tp import train_model

generate_hu_moments_file()
model = train_model()
load_and_test(model)
