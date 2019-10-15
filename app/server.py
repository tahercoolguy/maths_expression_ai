import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from io import BytesIO
import cv2
import os

export_file_url = 'https://www.dropbox.com/s/1vgceupaadsytoo/numbers_operation.pkl?dl=1'
export_file_name = 'numbers_operation.pkl'

classes = ['!',
  '(',
  ')',
  '+',
  '-',
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  '=',
  'div',
  'times']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))



async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


predictions=[]
def predict_append():
    im = cv2.imread("new.jpg")

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.cv2.THRESH_BINARY_INV)
    rotated_image=rotate_image(im_th,5)
    img=rotated_image


    # print(os.path.realpath(__file__))

    ctrs, hier  = cv2.findContours(rotated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours in the image
    # ctrs, hier ,_ = cv2.findContours(rotated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Get rectangles contains each contour
    # rects = [cv2.boundingRect(ctr) for ctr in hier]
    cntrs,bb=sort_contours(ctrs)

    for (i, c) in enumerate(cntrs):
        (x, y, w, h) = cv2.boundingRect(c)
    

        # WRITE YOUR CODE HERE!
        # use slicing and the (x, y, w, h) values of the bounding
        # box to create a subimage based on this contour
        y=y
        h=h
        w=w
        x=x


        if h > 80:
            subImg = rotated_image[y : y + h, x : x + w]
            roi = cv2.resize(subImg, (45, 45), interpolation=cv2.INTER_AREA)
            # roi = cv2.dilate(roi, (3, 3))
        
            # cv2_imshow(roi) #google path for cv2
            
            # WRITE YOUR CODE HERE!
            # save the subimage as sub-x.jpg, where x is the number
            # of this contour. HINT: try "sub-{0}".format(i) to 
            # create the filename
            cv2.imwrite(filename = "sub-{}.jpg".format(i), img = subImg)

            img_pl=PIL.Image.open("sub-{}.jpg".format(i)).convert('L')
            blurred_image = img_pl.filter(PIL.ImageFilter.BLUR)
            inverted_image = PIL.ImageOps.invert(blurred_image)

            threshold = 120
            im_th = inverted_image.point(lambda p: p > threshold and 255)
            im_resize = im_th.resize((45,45), PIL.Image.ANTIALIAS)
            im_cont = im_resize.filter(PIL.ImageFilter.CONTOUR)
            im_resize.save("new.jpg")
            m=open_image("new.jpg")
            predictions.append(str(learn.predict(m)[0]))

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    img.save("new.jpg")
    #new Code For Digits
    predict_append()
    #new Ends
    # prediction = learn.predict(img)[0]
    predict=""
    for i in predictions:
        if i == "times":
            predict=predict+" *"
        elif i == "div":
            predict=predict+" /"
        else :
            predict=predict+" "+str(i)
    
    try:
        predict = predict + " result " + str(eval(predict))
    except:
        predict="Kindly Enter proper Equation"
    predictions.clear()
    return JSONResponse({'result': str(predict)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

