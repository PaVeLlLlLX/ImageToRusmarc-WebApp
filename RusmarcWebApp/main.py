import os
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
#import numpy as np
import io
import time 
import base64
from model import ImageToRusmarcModel, generate_square_subsequent_mask 
from dataset import transform_test
from utils import decode_sequence

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

print("Загрузка модели...")
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './path/to/your/checkpoint.pth' 

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Чекпоинт не найден по пути: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

model = ImageToRusmarcModel(
    num_tokens=checkpoint['num_chars'],
    eos_token_id=checkpoint['eos_token_id'],
    sos_token_id=checkpoint['sos_token_id'],
    pad_token_id=checkpoint['pad_token_id']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"Модель загружена на {device}.")

print("Applying dynamic quantization to the model...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8)
print("Quantization finished.")

alphabet = checkpoint['alphabet']
sos_token_id = checkpoint['sos_token_id']
eos_token_id = checkpoint['eos_token_id']
pad_token_id = checkpoint['pad_token_id']

app = FastAPI(title="RUSMARC Card Recognition")
templates = Jinja2Templates(directory="templates") 
app.mount("/static", StaticFiles(directory="static"), name="static")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Отображает главную страницу с формой загрузки."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def handle_predict(request: Request, card_image: UploadFile = File(...)):
    if not allowed_file(card_image.filename):
        print(f"Ошибка: Недопустимый тип файла - {card_image.filename}")
        return templates.TemplateResponse("index.html", 
                                          {"request": request, 
                                           "error": "Недопустимый тип файла. Разрешены: png, jpg, jpeg, gif, bmp, tiff"})


    print(f"Получен файл: {card_image.filename}, тип: {card_image.content_type}")
    
    
    try:
        contents = await card_image.read()
        img_bytes_io = io.BytesIO(contents)

        img_bytes_io.seek(0) 
        image_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
        print("Изображение закодировано в Base64.")
 
        img_bytes_io.seek(0) 
        pil_image = Image.open(img_bytes_io).convert('RGB')
        print("Изображение прочитано для обработки.")

        start_transform = time.time()
        img_tensor: torch.Tensor = transform_test(pil_image).unsqueeze(0).to(device) 
        print(f"Трансформация заняла: {time.time() - start_transform:.4f} сек")

        print("Генерация предсказания...")
        start_predict = time.time()
        with torch.no_grad():
            pred_token_ids_raw = model.generate(img_tensor, max_len=200) 
            pred_token_ids = pred_token_ids_raw.cpu()[0] 
        print(f"Предсказание заняло: {time.time() - start_predict:.4f} сек")


        from torch.profiler import profile, record_function, ProfilerActivity

        n_runs = 10
        timings = []
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    _ = quantized_model.generate(img_tensor, max_len=200)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                
        print(f"Compiled model average time over {n_runs} runs: {timings} seconds")

    

        predicted_text = decode_sequence(pred_token_ids, alphabet, sos_token_id, eos_token_id, pad_token_id)
        print("Предсказание готово.")

        return templates.TemplateResponse("result.html", 
                                          {"request": request, 
                                           "prediction": predicted_text,
                                           "filename": card_image.filename,
                                           "image_base64": image_base64})

    except Exception as e:
        print(f"Ошибка обработки файла: {e}")
        return templates.TemplateResponse("index.html", 
                                          {"request": request, 
                                           "error": f"Ошибка при обработке изображения: {e}"})
    finally:
         await card_image.close()


if __name__ == "__main__":
    host =  "0.0.0.0"
    port = 5000
    uvicorn.run("main:app", host=host, port=port)