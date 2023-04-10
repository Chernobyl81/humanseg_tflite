import time
import cv2 as cv
import tflite_runtime.interpreter as tflite
import numpy as np


height = 480
width = 640

green_mat = np.zeros((height, width, 3), np.uint8)
green_mat[:] = (0, 255, 0)

def normalize(mat):
    mat /= 255.0
    mat -= 0.5
    mat /= 0.5

def infer(mat):
    interpreter = tflite.Interpreter(
    model_path='saved_model/ppseg_lite_portrait_398x224_with_softmax_fixed_float32.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print(input_details)
    print(output_details)
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], mat)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    return output_data


def postprocess(tensor, origin_img, bg):
    score_map = tensor[0, 1, :, :]

    mask_original = score_map.copy()
    mask_original = (mask_original * 255).astype("uint8")
    _, mask_thr = cv.threshold(mask_original, 240, 1,
                                cv.THRESH_BINARY)
    kernel_erode = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_CROSS, (25, 25))
    mask_erode = cv.erode(mask_thr, kernel_erode)
    mask_dilate = cv.dilate(mask_erode, kernel_dilate)
    score_map *= mask_dilate
   
    h, w, _ = origin_img.shape
    
    score_map = score_map[np.newaxis, np.newaxis, ...]
    print(f'scroe_map\'s shape {score_map.shape}')

    scrore_transposed = np.transpose(score_map.squeeze(1), [1, 2, 0])
    print(f'scrore_transposed\'s shape {scrore_transposed.shape}')
    
    tmp = cv.cvtColor(scrore_transposed, cv.COLOR_GRAY2BGR)
    tmp = cv.resize(tmp, (w, h))
    tmp = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
    print(f'tmp\'s shape {tmp.shape}')

    alpha = np.array(tmp)
    alpha = np.expand_dims(alpha, axis=-1)
    print(f'alpha\'s shape {alpha.shape}')

   
    bg = cv.resize(bg, (w, h))
    if bg.ndim == 2:
        bg = bg[..., np.newaxis]

    out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)
    return out


if __name__ == '__main__':
    image = cv.imread('images/glnz1.jpeg')
    resized_image = cv.resize(image, (398, 224)) 
    image_array = np.array(resized_image, dtype=np.float32)

    start_time = time.time()    

    normalize(image_array)
   
    input_tensor = np.expand_dims(image_array, axis=0)
    output_tensor = infer(input_tensor)

    result = postprocess(output_tensor, image, green_mat)

    end_time = time.time()
    print(f'inference time: {(end_time - start_time) * 1000:.6f} millseconds')
    cv.imwrite('result.jpeg', result)