from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import functions as F
import torch
import base64




app = Flask(__name__)
CORS(app)

## Initializations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
binary_classes = ("Hatasız" , "Hatalı")
errorTypes = ("Baş Aşağı","Baş Yukarı","Çentiğin Önden ısırılmış","Çentiğin Arkadan ısırılmış",
              "Hasta başını sağa sola çevirmiş","Hasta başını sağa sola yatırmış")

## Models paths.
binary_rontgen_modelPath = "Models/binaryCls_butun_rontgen.pth"
binary_mandibular_modelPath = "Models/binaryCls_mandibular.pth"
multiCls_rontgen_modelPath = "Models/multiCls_butun_rontgen.pth"
multiCls_mandibular_modelPath = "Models/multiCls_mandibular.pth"
mask_rcnn_modelPath = "Models/model.pt"            ## Mask R-CNN model file path.
excel_path = "labels/Total_Excel.xlsx"             ## True labels excel file path.

## Required to load pretrained deit transformer model.
binary_label2id, binary_id2label = F.initModelArgs(binary_classes)
multi_label2id, multi_id2label = F.initModelArgs(errorTypes)


@app.route('/')
def home():
    return render_template('index.html')

## Handle Data sent by client side.
@app.route('/uploadImage',methods=['POST'])
def uploadImage():
    global mask_rcnn_model, device
    if request.method == 'POST':
        result = {}
        if request.files['rontgen']== '':
            result['response'] = "Empty request"
        else:
            image = request.files['rontgen']
            if image.filename != '':
                ##print(image.filename)
                ## Load mask r-cnn model.
                mask_rcnn_model = F.create_model(mask_rcnn_modelPath,device).to(device)
                mask_rcnn_result = F.mask_rcnn_model_prediction(image,mask_rcnn_model,device)
                mask_rcnn_model = None

                ## mask r-cnn results.
                overlay_mask = F.draw_overlay_mask(image,mask_rcnn_result)
                mandibula = F.extract_mask(image,mask_rcnn_result)
                overlay_mask_bytes = F.getBytesObject(overlay_mask)
                mandibula_bytes = F.getBytesObject(mandibula)
                if overlay_mask_bytes is None or mandibula_bytes is None:
                    result['response'] = 'Error in processing data'
                else:
                    overlay_mask_data = base64.b64encode(overlay_mask_bytes.getvalue()).decode('utf-8')           ## convert overlay image to bytes arrray.
                    mandibula_data = base64.b64encode(mandibula_bytes.getvalue()).decode('utf-8')                 ## convert mask to bytes arrray.
                    result = {'response' : "Success processing result",'mandibula' : mandibula_data, "overlayMask" : overlay_mask_data}

                    ## Start of classification method.
                    model = F.initModel(binary_id2label,binary_label2id,device,binary_rontgen_modelPath)
                    binary_rontgen_res = F.predict(model,device,image)
                    result['binary_rontgen_result'] = binary_classes[binary_rontgen_res['prediction']]

                    model = None
                    model = F.initModel(binary_id2label,binary_label2id,device,binary_mandibular_modelPath)
                    binary_mandibular_res = F.predict(model,device,image)
                    result['binary_mandibular_result'] = binary_classes[binary_mandibular_res['prediction']]

                    binary_combine_res = F.combine(binary_rontgen_res['prob'],binary_mandibular_res['prob'])
                    result['binary_combine_result'] = binary_classes[binary_combine_res]

                    model = None
                    model = F.initModel(multi_id2label,multi_label2id,device,multiCls_mandibular_modelPath)
                    multiCls_mandibular_res = F.predict(model,device,image)

                    model = None
                    model = F.initModel(multi_id2label,multi_label2id,device,multiCls_rontgen_modelPath)
                    multiCls_rontgen_res = F.predict(model,device,image)
                    model = None
                    
                    multiCls_combine_res = F.combine(multiCls_rontgen_res['prob'],multiCls_mandibular_res['prob'])

                    if binary_rontgen_res['prediction'] == 0:
                        result['multiCls_rontgen_res'] = "Yok"
                    else:
                        result['multiCls_rontgen_res'] = errorTypes[multiCls_rontgen_res['prediction']] 

                    if binary_mandibular_res['prediction'] == 0:
                        result['multiCls_mandibular_res'] = "Yok"
                    else:
                        result['multiCls_mandibular_res'] = errorTypes[multiCls_mandibular_res['prediction']]

                    if binary_combine_res == 0:
                        result['multiCls_combine_res'] = "Yok"
                    else:
                        result['multiCls_combine_res'] = errorTypes[multiCls_combine_res]

                    true1, true2 = F.getTrueLabels(excel_path,image.filename)
                    if true1 == -1:
                        result['trueError'] = ""
                        result['trueErrorType'] = ""
                    else:
                        result['trueError'] = binary_classes[true1]
                        if true2 == '' or true2 == 0:
                            result['trueErrorType'] = "-"
                            
                        else:
                            result['trueErrorType'] = errorTypes[true2-1]

                ## ----------------------
            else:
                result['response'] = 'file not found'
        
        return jsonify(result)                      ## Send results to client side.
    
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)