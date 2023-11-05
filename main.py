import io
import math
from fastapi import FastAPI, Response, UploadFile, File
from PIL import Image
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


def find_nearest_factors(n):
    factor1 = int(math.sqrt(n))
    while n % factor1 != 0:
        factor1 -= 1
    factor2 = n // factor1
    return factor1, factor2


@app.post("/image_encode/")
async def convert_binary_to_image(image_file: UploadFile = File(...)):
    data = await image_file.read()

    n: int = len(data)
    m: int = 8
    arr: np.ndarray = np.zeros((n, m), dtype=np.uint8)

    counter: int = 0
    for byte in data:
        counter += 1
        print(f"{counter:04}: {byte} -> {bin(byte)[2:].zfill(m)}")
        arr[counter - 1, :] = [int(bit) for bit in bin(byte)[2:].zfill(m)]

    # converting from np array to image
    bw_arr: np.ndarray = np.where(arr == 0, 0, 255)

    # reshaping to nearest factors to make it square
    n, m = find_nearest_factors(n * m)
    bw_arr = bw_arr.reshape((n, m))

    img = Image.fromarray(bw_arr.astype(np.uint8))
    img.save("encoded.jpg")
    return Response(content=img.tobytes(), media_type="image/jpeg")


@app.post("/image_decode")
async def convert_(image_file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await image_file.read()))

    image_arr = np.array(image)
    bw_arr: np.ndarray = np.where(image_arr >= 100, 1, 0)
    n, m = bw_arr.shape
    n = n * m
    m = 8

    bw_arr = bw_arr.flatten()
    binary_str: str = "".join(map(str, bw_arr))
    len_bytes: int = len(binary_str) // 8
    data = int(binary_str, 2).to_bytes(length=len_bytes, byteorder='big')
    with open("decoded.jpg", "wb") as f:
        f.write(data)
    return Response(content=data, media_type="image/jpeg")
