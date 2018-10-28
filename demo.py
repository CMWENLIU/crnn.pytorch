import os
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn

image_dir = 'data/images'
image_paths = os.listdir(image_dir)
for i in image_paths:
  image = os.path.join(image_dir, i)
model_path = './data/crnn.pth'
#img_path = './data/0.12.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
with open('result.html', 'w') as outf:
  with open('htmlhead.txt', 'r') as fh:
    for line in fh:
      outf.write(line)
  imghead = '<img src="'
  imgtail = '">'
  for i in image_paths:
    img_path = os.path.join(image_dir, i)
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s:%-20s => %-20s' % (i, raw_pred, sim_pred))
    outf.write(imghead + img_path + imgtail)
    outf.write('<p style="color:blue;font-size:46px;">' + sim_pred + '</p>' + '<hr>')
    with open('htmltail.txt', 'r') as ft:
      for line in ft:
        outf.write(line)
