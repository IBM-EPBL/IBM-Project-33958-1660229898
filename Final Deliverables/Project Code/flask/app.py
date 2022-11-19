from flask import Flask,render_template,request
import cv2
import numpy as np
import io
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("fruit.h5")
model2 = keras.models.load_model("vegetable.h5")
categories = ['Apple___Black_rot','Apple___healthy','Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Peach___Bacterial_spot','Peach___healthy']
categories2 = ['Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight','Tomato___Bacterial_spot','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot']


@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/fruit',methods=['GET', 'POST'])
def fruit():
    return render_template('fruit.html')

@app.route('/vegge',methods=['GET', 'POST'])
def vegge():
    return render_template('vegetable.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            image = request.files['image']
            
            # idata = base64.b64encode(image.read()).decode('utf-8')
            in_memory_file = io.BytesIO()
            image.save(in_memory_file)
            data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
            color_image_flag = 1
            img = cv2.imdecode(data, color_image_flag)
        except:
            print('Upload an Image')
            return render_template("predict.html")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(100,100))
        img = np.reshape(img,[1,100,100,3])
        img = np.array(img, dtype=np.float32)
        prediction = np.argmax(model.predict(img),axis=1)
        print(prediction)
        res = categories[prediction[0]]
        if prediction[0]==0:
            idata = 'Captan and Sulfur Fertilizer'
        elif prediction[0]==3:
            idata = 'Trichoderma Harzianum or Bacillus subtilis'
        elif prediction[0]==4:
            idata = 'Oxytetracycline (Mycoshield and generic equivalents), and syllit+captan'
        else:
            idata = "The Leaf is Healthy,  No Fertilizer Needed"
        return render_template("predict.html",res=res,idata=idata)
    return render_template("predict.html")

@app.route('/predict2',methods=['GET', 'POST'])
def predict2():
    if request.method == "POST":
        try:
            image = request.files['image']
            in_memory_file = io.BytesIO()
            image.save(in_memory_file)
            data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
            color_image_flag = 1
            img = cv2.imdecode(data, color_image_flag)
        except:
            print('Upload an Image')
            return render_template("predict2.html")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(100,100))
        img = np.reshape(img,[1,100,100,3])
        img = np.array(img, dtype=np.float32)
        prediction = np.argmax(model2.predict(img),axis=1)
        print(prediction)
        res = categories2[prediction[0]]
        if prediction[0]==0:
            idata = 'Acibenzolar-S-methyl, ALS'
        elif prediction[0]==2:
            idata = 'Mancozeb and chlorothalonil'
        elif prediction[0]==4:
            idata = 'Dithane (mancozeb) MZ'
        elif prediction[0]==5:
            idata = 'Azoxystrobin or Penthiopyrad'
        elif prediction[0]==6:
            idata = 'Nitrogen fertilizer'
        elif prediction[0]==7:
            idata = 'Apple-cider and vinegar Spray'
        elif prediction[0]==8:
            idata = 'Copper fungicide'
        else:
            idata = "The Leaf is Healthy, No Fertilizer Needed"
        print(res)
        return render_template("predict2.html",res=res,idata=idata)
    return render_template("predict2.html")
    
 
if __name__ == '__main__':
    app.run(debug=True,port=5000,host="0.0.0.0")