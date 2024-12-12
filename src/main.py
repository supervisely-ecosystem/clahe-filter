from js import ImageData, Object, slyApp
from pyodide.ffi import create_proxy
import numpy as np
import cv2


def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

def main(mode='process'):

  def clahe(img_rgb_8_bits, clipLimit=2.0, tileGridSize=(8, 8)):
    img_lab = cv2.cvtColor(img_rgb_8_bits, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab_planes[0] = clahe.apply(lab_planes[0])
    img_lab = cv2.merge(lab_planes)
    img_rgb_8_bits = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return img_rgb_8_bits
  
  app = slyApp.app
  store = slyApp.store
  app = getattr(app, '$children')[0]

  context = app.context
  state = app.state

  # store action example
  # appEventEmitter = app.appEventEmitter
  # eventData = Object()
  # eventData.action = 'videos/nextImage'
  # eventData.payload = {}
  # appEventEmitter.emit('store-action', eventData)

  cur_img = getattr(store.state.videos.all, str(context.imageId))
  img_src = cur_img.sources[0]
  img_cvs = img_src.imageData

  img_ctx = img_cvs.getContext("2d")

  if state.imagePixelsDataImageId != context.imageId:
    img_data = img_ctx.getImageData(0, 0, img_cvs.width, img_cvs.height).data

    # reshape flat array of rgba to numpy
    state.imagePixelsData = np.array(img_data, dtype=np.uint8).reshape(img_cvs.height, img_cvs.width, 4)
    state.imagePixelsDataImageId = context.imageId


  new_img_data = None
  img_arr = state.imagePixelsData

  if mode == 'restore':
    new_img_data = img_arr.flatten()
  else:
    clip_limit = state.SliderAutoId6MqE3.value
    if state.labCheck is False:
      img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2GRAY)
      clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

      cl_img_gray = clahe.apply(img_gray)
      cl_img_rgb = cv2.cvtColor(cl_img_gray, cv2.COLOR_GRAY2RGB)

      alpha_channel = img_arr[:, :, 3]
      cl_img_rgba = np.dstack((cl_img_rgb, alpha_channel))

      new_img_data = cl_img_rgba.flatten().astype(np.uint8)
    else:
      new_img_data = clahe(img_arr, clip_limit, (8, 8)).flatten().astype(np.uint8)

  pixels_proxy = create_proxy(new_img_data)
  pixels_buf = pixels_proxy.getBuffer("u8clamped")
  new_img_data = ImageData.new(pixels_buf.data, img_cvs.width, img_cvs.height)

  img_ctx.putImageData(new_img_data, 0, 0)
  img_src.version += 1

  pixels_proxy.destroy()
  pixels_buf.release()

main
