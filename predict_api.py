from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from pydantic import BaseModel
import os
import random
import string
from main import preprocess_image, generate_and_save_gradcam, save_imposed_image

app = FastAPI()



class ImageData(BaseModel):
    image: list

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Grad-CAM Visualization Project"}

@app.post("/predict")
async def explain_features(file: UploadFile = File(...)):

    if not file:
        return {"message": "No Upload file sent"}
    else: 
        try:
            image = await file.read()
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            upload_dir = os.path.join(os.path.dirname(__file__),  'artifacts')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, f"image_{random_string}.jpg")
            with open(file_path, "wb") as f:
                f.write(image)
            processed_image = preprocess_image(file_path)
            model = VGG16(weights='imagenet')
            last_conv_layer_name = "block5_conv3"
            superimposed_img = generate_and_save_gradcam(file_path, model, last_conv_layer_name)
            preds = model.predict(processed_image) 
            image_name = f"gradcam_{random_string}.jpg"
            gradcam_image_path = os.path.join(upload_dir, image_name)
            save_imposed_image(gradcam_image_path,superimposed_img)
        except FileNotFoundError as e:
            return {"error": f"[Errno 2] No such file or directory: '{file_path}'"}
        except Exception as e:
            return {"error": str(e)}
    
    return {"Predicted": decode_predictions(preds, top=1)[0][0][1],
            "Grad_CAM_Image": image_name},FileResponse(gradcam_image_path, media_type="image/jpeg", filename=image_name)