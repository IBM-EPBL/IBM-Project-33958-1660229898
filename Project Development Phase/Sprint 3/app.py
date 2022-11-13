import base64
from flask import Flask,render_template,request
import cv2
import numpy as np
import io
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("F:\\project ibm\\flask\\fruit.h5")
model2 = keras.models.load_model("F:\\project ibm\\flask\\vegetable.h5")
categories = ['Apple___Black_rot','Apple___healthy','Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Peach___Bacterial_spot','Peach___healthy']
categories2 = ['Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight','Tomato___Bacterial_spot','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot']
 

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        image = request.files['image']
        # idata = base64.b64encode(image.read()).decode('utf-8')
        in_memory_file = io.BytesIO()
        image.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(100,100))
        img = np.reshape(img,[1,100,100,3])
        img = np.array(img, dtype=np.float32)
        prediction = np.argmax(model.predict(img),axis=1)
        print(prediction)
        res = categories[prediction[0]]
        print(res)
        return render_template("predict.html",res=res,idata='idata')
    return render_template("predict.html")

@app.route('/predict2',methods=['GET', 'POST'])
def predict2():
    if request.method == "POST":
        image = request.files['image']
        in_memory_file = io.BytesIO()
        image.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(100,100))
        img = np.reshape(img,[1,100,100,3])
        img = np.array(img, dtype=np.float32)
        prediction = np.argmax(model2.predict(img),axis=1)
        print(prediction)
        res = categories2[prediction[0]]
        print(res)
        return render_template("predict2.html",res=res,idata='idata')
    return render_template("predict2.html")
    
 
if __name__ == '__main__':
    app.run()