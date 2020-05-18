import pandas as pd
import numpy as np
import json
import argparse
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

batch_size = 102
img_size = 224

def predict(img_path, model, top_k, class_names):
    image = Image.open(img_path)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (img_size, img_size))
    image /= 255
    image = image.numpy()
    list = model.predict(image)
    prob = []
    classes = []
    rank = list[0].argsort()[::-1]
    for i in range(top_k):
        index = rank[i] + 1
        c = class_names[str(index)]
        prob.append(list[0][index])
        classes.append(c)
    return prob, classes
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('model')
    parser.add_argument("--top_k", required = False, default = 3)
    parser.add_argument("--category_names", required = False, default = "label_map.json")
    args = parser.parse_args()
    print(args)
    model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer} )
    class_names = {}
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    prob, classes = predict(args.img_path, model, args.top_k, class_names)
    print(prob)
    print(classes)

if __name__ == '__main__': main()
