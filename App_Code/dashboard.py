import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import os
import torch
import cv2
import numpy as np
import glob as glob
# from xml.etree import ElementTree as et
from tqdm.auto import tqdm
import torch
import io
from PIL import Image
import dash_bootstrap_components as dbc
from helpers import load_model, predict_label, crop_components, read_xml, load_img, predict_box, create_model, align_images
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

boxtype = 'no box detected'

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Quality Control Dashboard"),
    html.Hr(),
    html.H3("Step 1 : Upload image: "),
    dcc.Upload(
        id='upload-image',
        children=dbc.Button('Select Image', color="primary"),
        multiple=False,
        accept='image/*'
    ),
    html.Br(),
    html.H3("Step 2: Show image and box type"),
    dbc.Button('Show Image and Box Type',
               color="primary", id='show-image-button'),
    html.Br(),
    html.Div(id='output-image'),
    html.Br(),
    html.H3("Step 3: Predict bounding box, align and show cropped image: "),
    dbc.Button("Show bounding box", id="bounding-box-button", color="primary"),
    html.Br(),
    html.Div(id="bounding-box-output"),
    html.Br(),
    html.H3("Step 4: Classify hardware components"),
    dbc.Button("Perform Classification",
               id="classification-button", color="primary"),
    html.Br(),
    html.Div(id="classification-output"),
    html.Br()
])


def classify_hardware_elements():

    xml_dict = read_xml(boxtype)

    cropped, labels = crop_components(
        xml_dict)
    show = []

    # loop over all components
    for i in range(len(cropped)):

        # write component jpg
        cv2.imwrite(f'temp_img/1/{labels[i]}.jpg', cropped[i])

        # load model
        model = load_model(labels[i])

        # load image
        dataloader = load_img('temp_img')

        # predict label
        pred = predict_label(model, dataloader)

        outputs_list = []
        if pred == 1:
            outputs_list.append(f'{labels[i]} successfully detected. ')
        else:
            outputs_list.append(f'{labels[i]} is missing. Please inspect! ')
        outputs_list.append(f'---------------------------------')

        # delete jpg
        os.chdir('temp_img/1')
        for file in os.listdir():
            os.remove(file)
        os.chdir('../../')

        div_list = [html.Div(
            html.H4(data)
        ) for data in outputs_list]

        div_list2 = html.Div(
            html.Img(src='data:image/jpeg;base64,{}'.format(image_to_base64(cropped[i])), width="100", height="100"))

        to_show = html.Div(children=[div_list2, html.Div(children=div_list)])
        show.append(to_show)

    return html.Div(children=show)


def parse_contents(filename):
    if filename is None:
        return html.Div()
    else:
        with open(os.path.abspath(filename), 'rb') as f:
            data = f.read()
        abs_filepath = os.path.abspath(filename)
        encoded_image = base64.b64encode(data).decode()

        shutil.copy(filename, 'temp_img/1/box.jpg')
        global boxtype
        boxtype = predict_box()

        return html.Div([
            html.Br(),
            html.Img(src='data:image/jpg;base64,{}'.format(encoded_image),
                     width="600", height="400"),
            html.H5(f'Box type: {boxtype}', style={
                "font-style": "italic"
            })
        ])


def image_to_base64(image):
    with io.BytesIO() as buffer:
        Image.fromarray(image).save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def perform_bb_recognition(filename):
    model = create_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(
        f'models/{boxtype}.pth', map_location=device
    ))
    model.eval()

    detection_threshold = 0.8
    CLASSES = ['background', 'Allen-Bradley']

    bb_pred_list = []
    image_name = filename.split('/')[-1].split('.')[0]
    image = cv2.imread(filename)
    orig_image = image.copy()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0

    image = np.transpose(image, (2, 0, 1)).astype(np.float)

    image = torch.tensor(image, dtype=torch.float)

    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) != 0:

        bb_pred_tensor = outputs[0]['boxes'][0]
        bb_pred_list += torch.Tensor.tolist(bb_pred_tensor)
        bb_pred_tensor = bb_pred_tensor.unsqueeze(0)

        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()

        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            '''
            cv2.rectangle(orig_image,
                        (int(bb_truth[0]), int(bb_truth[1])),
                        (int(bb_truth[2]), int(bb_truth[3])),
                        (255, 0, 0), 2)
             '''
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)

        img_cropped = orig_image[box[1]:box[3], box[0]:box[2]]
        template = cv2.imread(f"templates/{boxtype}.jpg")
        img_cropped_aligned = (align_images(img_cropped, template))

        cv2.imwrite('temp_img/1/img_cropped_aligned.jpg', img_cropped_aligned)

    return html.Div([
        html.Br(),
        html.Img(src='data:image/jpeg;base64,{}'.format(
            image_to_base64(img_cropped_aligned)), width="700", height="300"),
        html.H4("------Box Prediction completed-------", style={
            "font-style": "italic"
        })
    ])


@app.callback(Output('output-image', 'children'),
              [Input('show-image-button', 'n_clicks')],
              [State('upload-image', 'filename')])
def update_output(n_clicks, filename):
    if n_clicks is not None and filename is not None:
        return parse_contents(filename)


@app.callback(Output('bounding-box-output', 'children'),
              [Input('bounding-box-button', 'n_clicks')],
              [State('upload-image', 'filename')])
def update_output(n_clicks, filename):
    if n_clicks is not None and filename is not None:
        return perform_bb_recognition(filename)


@app.callback(Output('classification-output', 'children'),
              [Input('classification-button', 'n_clicks')],
              [State('upload-image', 'filename')])
def update_output(n_clicks, filename):
    if n_clicks is not None and filename is not None:
        return classify_hardware_elements()


if __name__ == '__main__':
    app.run_server(debug=True, port=2002)
