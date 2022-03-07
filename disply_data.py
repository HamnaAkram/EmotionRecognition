from IPython.core.display import display, HTML
from PIL import Image
import io
import base64
import pandas as pd

img = Image.open("/home/hamna/PycharmProjects/EmotionRecognition/testdata/contempt/0.jpg")
img_buffer = io.StringIO()
img.save(img_buffer, format="PNG")
imgStr = base64.b64encode(img_buffer.getvalue())

data = pd.DataFrame({"A":[1,2,3,4,5], "B":[10,20,30,40,50]})
data.loc[:,'img'] = '<img src="data:image/png;base64,{0:s}">'
html_all = data.to_html(escape=False).format(imgStr)
display(HTML(html_all))
