from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import pandas as pd
extractor = ViTFeatureExtractor.from_pretrained("nateraw/food")
model = ViTForImageClassification.from_pretrained("nateraw/food")
dataset = pd.read_excel('../datasets/database.xlsx', skiprows=1)
unhealty = set('apple_pie', 
             'baby_back_ribs',
             'grilled_cheese_sandwich',
             'baklava',
             'carot_cake',
             'beef_carpaccio',
             'beignets',
             'bread_pudding',
             'breakfast_burrito',
             'cannoli',
             'carrot_cake',
             'cheesecake',
             'chicken_wings',
             'chicken_quesadilla',
             'chocolate_cake',
             'chocolate_mousse',
             'churros',
             'creme_brulee',
             'croque_madame',
             'cup_cakes',
             'donuts',
             'filet_mignon',
             'french_fries',
             'grilled_cheese_sandwich',
             'hamburger',
             'hot_dog',
             'ice_cream',
             'macaroni_and_cheese',
             'macarons',
             'nachos',
             'onion_rings',
             'pancakes',
             'panna_cotta',
             'peking_duck',
             'pizza',
             'poutine',
             'prime_rib',
             'pulled_work_sandwich',
             'red_velvet_cake',
             'steak',
             'strawberry_shortcake',
             'tacos',
             'tiramisu',
             'waffles',
             'beef_tartare',
             'pork_chop',
             'pulled_pork_sandwich'
             )
def predict(image):
    input = extractor(images=image, return_tensors='pt')
    output = model(**input)
    logits = output.logits
    
    pred_class = logits.argmax(-1).item()
    return model.config.id2label[pred_class]

def check_food(food, counter):
    ret_dict = dict()
    ret_dict['food_pred'] = None
    ret_dict['cal'] = None
    ret_dict['carb'] = None
    ret_dict['prot'] = None
    ret_dict['fat'] = None
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
        ret_dict['food_pred'] = food_info["Main food description"].iloc[0]
        ret_dict['cal'] = food_info['Energy (kcal)'].iloc[0]
        ret_dict['carb'] = food_info['Carbohydrate (g)'].iloc[0]
        ret_dict['prot'] = food_info['Protein (g)'].iloc[0]
        ret_dict['fat'] = food_info['Total Fat (g)'].iloc[0]
    return ret_dict
def get_cc(img_class, weight):
    food = img_class.split('_')
    if food[-1][-1] == "s":
        food[-1] = food[-1][:-1]
    food_dict['food_pred'] = None
    counter = 0
    while (not food_dict['food_pred']) &  (counter <= 4):
        food_dict = check_food(food,counter)
        counter += 1
    if not food_dict['food_pred']:
        food_dict = {'Error': "Food not found"}
    return food_dict
def count_calories (img_path, weight):
    image = Image.open(img_path)
    pred = predict(image)
    output = get_cc(pred, weight)
    output['healthy_food'] = pred not in unhealty
    return (pred, output)