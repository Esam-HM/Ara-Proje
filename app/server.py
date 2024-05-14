from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import functions as F
import torch
import base64



app = Flask(__name__)
CORS(app)

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
mask_rcnn_model = F.create_model("Models/model.pt",device).to(device)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/uploadImage',methods=['POST'])
def uploadImage():
    global mask_rcnn_model, device
    if request.method == 'POST':
        result = {}
        if request.files['rontgen'] == '':
            result['response'] = "Empty request"
        else:
            image = request.files['rontgen']
            if image.filename != '':

                ## mask r-cnn results.
                mask_rcnn_result = F.mask_rcnn_model_prediction(image,mask_rcnn_model,device)
                overlay_mask = F.draw_overlay_mask(image,mask_rcnn_result)
                mandibula = F.extract_mask(image,mask_rcnn_result)
                overlay_mask_bytes = F.getBytesObject(overlay_mask)
                mandibula_bytes = F.getBytesObject(mandibula)
                if overlay_mask_bytes is None or mandibula_bytes is None:
                    result['response'] = 'Error in processing data'
                else:
                    overlay_mask_data = base64.b64encode(overlay_mask_bytes.getvalue()).decode('utf-8')
                    mandibula_data = base64.b64encode(mandibula_bytes.getvalue()).decode('utf-8')
                    result = {'response' : "Success processing result",'mandibula' : mandibula_data, "overlayMask" : overlay_mask_data}
                ## ----------------------
            else:
                result['response'] = 'file not found'  

        return jsonify(result)
    
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)