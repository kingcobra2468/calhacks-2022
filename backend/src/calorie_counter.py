from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import pandas as pd
extractor = ViTFeatureExtractor.from_pretrained("nateraw/food")
model = ViTForImageClassification.from_pretrained("nateraw/food")
dataset = pd.read_excel('../datasets/database.xlsx', skiprows=1)

def predict(image):
    input = extractor(images=image, return_tensors='pt')
    output = model(**input)
    logits = output.logits
    
    pred_class = logits.argmax(-1).item()
    return model.config.id2label[pred_class]

def check_food(food, counter):
    foodPred, cal, carb, prot, fat = None, None, None, None, None
    if counter == 0:
        if len(food) >= 3:
            foodName = food[0].capitalize() + " " + food[1] + " " + food[2] + ","
        elif len(food) == 2:
            foodName = food[0].capitalize() + " " + food[1] + ","
        elif len(food) == 1:
            foodName = food[0].capitalize() + ","
    elif counter == 1:
        if len(food) >= 3:
            foodName = food[0].capitalize() + " " + food[1] + " " + food[2]
        elif len(food) == 2:
            foodName = food[0].capitalize() + " " + food[1]
        elif len(food) == 1:
            foodName = food[0].capitalize()
    elif counter == 2:
        if len(food) >= 3:
            foodName = food[0] + " " + food[1] + " " + food[2]
        elif len(food) == 2:
            foodName = food[0] + " " + food[1]
        elif len(food) == 1:
            foodName = food[0]
    else:
        foodName = food[0].capitalize()
    food_info = dataset[dataset['Main food description'].apply(lambda x: x[:len(foodName)] == foodName)]
    condition = len(food_info) != 0
    if condition:
        foodPred = food_info["Main food description"].iloc[0]
        cal = food_info['Energy (kcal)'].iloc[0]
        carb = food_info['Carbohydrate (g)'].iloc[0]
        prot = food_info['Protein (g)'].iloc[0]
        fat = food_info['Total Fat (g)'].iloc[0]
    return foodPred, cal, carb, prot, fat
def get_cc(img_class, weight):
    food = img_class.split('_')
    if food[-1][-1] == "s":
        food[-1] = food[-1][:-1]

    foodPred, cal, carb, prot, fat = None, None, None, None, None
    counter = 0
    while (not foodPred) &  (counter <= 4):
        foodPred, cal, carb, prot, fat = check_food(food,counter)
        counter += 1
    if food:        
        output = foodPred + "\nCalories: " + str(round(cal * weight)/100) + " kJ\nCarbohydrate: " + str(round(carb * weight)/100) + " g\nProtein: " + str(round(prot * weight)/100) + " g\nTotal Fat: " + str(round(fat * weight)/100) + " g"      
    elif not food:
        output = "Food not found"
    return output
def count_calories (img_path, weight):
    image = Image.open(img_path)
    pred = predict(image)
    output = get_cc(pred, weight)
    return (pred, output)