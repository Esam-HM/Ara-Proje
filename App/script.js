let imageReference;  // To store selected image object file.
let flag = 0;        //To store display state of uploadDirContainer.
let control = 0;     // To prevent load image after clicking check button.

// Add listener to upload image button.
// It will click input button when upload button clicked.
document.getElementById('uploadBtn').addEventListener('click',function(){
    if(control==1){    // return if check button clicked.
        window.alert("Kontrol işlemi başladığı için resim seçilemez. İşlem bitince yeni resim seçebilirsiniz.");
        return;
    }
    document.getElementById('input').click();
});

// Handle selected image file.
document.getElementById('input').addEventListener('change', function(event) {
    loadImage(event.target.files[0]);
});

// Add listener to upload directory button.
// It will click fileInput button if upload button clicked.
document.getElementById('uploadDirBtn').addEventListener('click', function(){
    document.getElementById('fileInput').click();
});

// Add event listener to handle selected directory.
// When triggered it calls handleDirectory function.
document.getElementById('fileInput').addEventListener('change', handleDirectory);

// Add listener to handle image drop operation on upload container.
// When triggered it calls handleImageDropOperation function.
document.getElementsByClassName('uploadImageContainer')[0].addEventListener('drop', handleImageDropOperation);

// Add listener to drag over upload container event.
// It prevents drowser default operation.
document.getElementsByClassName('uploadImageContainer')[0].addEventListener('dragover', function(event) {
    event.preventDefault();
});

// Add event listener to checkBtn button.
// When clicked it calls startControlOperation function.
document.getElementById('checkBtn').addEventListener('click', startControlOperation);

// Add event listener to checkBtn button.
// when clicked, it returns to home page.
document.getElementById('backBtn').addEventListener('click',function(){
    const resCont = document.getElementById('resContId');
    const uploadCont = document.getElementById('uploadContId');

    control=0;
    document.getElementById('checkBtn').style.display = 'flex';
    resCont.style.display = 'none';
    uploadCont.style.display = 'flex';
    if(flag === 1){
        document.getElementById('uploadDirContainerId').style.display = 'flex';
    }

});

// To start control for uploaded image.
// Sends image to server side and waits the results.
// When results received, it prints them to UI.
function startControlOperation(){
    const progressBar = document.getElementById('progress');
    const img = document.getElementById('radiograph');
    const resultCont = document.getElementById('resContId');
    const inImage = document.getElementById('inputImg');
    const mand = document.getElementById('mask');
    const overlay= document.getElementById('overlayMask');
    const uploadCont = document.getElementById('uploadContId');
    const formData = new FormData();

    // return if no file selected.
    if(img.getAttribute('src') == ''){
        window.alert("Bir röntgen görüntüsü yüklemeniz gerekir.");
        return;
    }else{
        this.style.display = 'none';
    }

    progressBar.style.display = 'flex';
    
    if(imageReference == ''){                     // if selected image reference not found.
        progressBar.style.display = 'none';
        this.style.display = 'flex';
        return;
    }

    formData.append('rontgen',imageReference);    // Data to be sent to server side.
    control=1;
    // send data to server side.
    fetch("/uploadImage", {
        method: "POST",
        body: formData,
    }).then(response => {                        // Wait for server respose.
        if (!response.ok) {                      // Handles errors if happened.
            console.log('Error happend when getting response');
            progressBar.style.display = 'none';
            this.style.display = 'flex';
            control=0;
        }
        return response.json();
    }).then(data => {
        //console.log(data.response);
        inImage.src = img.getAttribute('src');
        mand.src = "data:image/png;base64," + data.mandibula;
        overlay.src = "data:image/png;base64," + data.overlayMask;
        document.getElementById('state1').textContent = "Durum: " + data.binary_rontgen_result;
        document.getElementById('state2').textContent = "Durum: " + data.binary_mandibular_result;
        document.getElementById('state3').textContent = "Durum: " + data.binary_combine_result;
        document.getElementById('errorType1').textContent = "Hata Tipi: " + data.multiCls_rontgen_res;
        document.getElementById('errorType2').textContent = "Hata Tipi: " + data.multiCls_mandibular_res;
        document.getElementById('errorType3').textContent = "Hata Tipi: " + data.multiCls_combine_res;

        if (data.trueError==''){
            document.getElementById('trueState').textContent = "Gerçek Durum: - ";
        }else{
            document.getElementById('trueState').textContent = "Gerçek Durum: " + data.trueError + " , Hata Tipi: " + data.trueErrorType;
        }
        
        // Controls for UI.
        progressBar.style.display = 'none';
        uploadCont.style.display = 'none';
        resultCont.style.display = 'flex';
        
        if (flag == 1){
            document.getElementById('uploadDirContainerId').style.display = 'none';  
        }
        console.log('Process finished');
    }).catch(error => {
        console.error('Error Happend with received data', error);
        progressBar.style.display = 'none';
        this.style.display = 'flex';
        control=0;
    });
}

// Load selected image to container.
function loadImage(file){
    const reader = new FileReader();
    const img = document.getElementById('radiograph');
    const iconContainer = document.getElementById('iconCont');
    
    reader.onload = function(e) {
        iconContainer.style.display = 'none';
        img.style.display = 'block';
        img.src = e.target.result;
        imageReference = file;
    };
    reader.readAsDataURL(file);
}

// Handle files list.
// Add all images from selected directory to list. 
function handleDirectory(event) {
    const fileList = document.getElementById('fileList');
    const container = document.getElementById('uploadDirContainerId');

    container.style.display = 'flex';
    const files = event.target.files;

    if (files && files.length > 0){
       fileList.innerHTML = ''; 
       flag = 1;
    }
  
    for (let i = 0, file; file = files[i]; i++) {
      if (file.type.startsWith('image/')){ 
        const listItem = document.createElement('div');
        const icon = document.createElement('img');
        const fileName = document.createElement('span');

        listItem.className = 'file-item';
        listItem.setAttribute('file_name',file.name);
        icon.src = URL.createObjectURL(file);
        icon.className = 'file-icon';
        icon.addEventListener('dragstart',preventIconDrag);
        listItem.addEventListener('click',selectItem);
        
        fileName.textContent = file.name;
        listItem.appendChild(icon);
        listItem.appendChild(fileName);
        fileList.appendChild(listItem);
      }
    }
    if(document.getElementById('resContId').style.display == 'flex'){
        container.style.display = 'none';
    }
}

// Get mime type of image.
// Used to convert image to file object.
function getMimeTypeFromExtension(filename) {
    const extension = filename.split('.').pop().toLowerCase();
    switch (extension) {
        case 'jpg':
            return 'image/jpg';
        case 'jpeg':
            return 'image/jpeg';
        case 'png':
            return 'image/png';
        default:
            return 'application/octet-stream'; //Default
    }
}

// Function to convert image URL to File object
function imageURLtoFile(url, filename){
    const mimeType = getMimeTypeFromExtension(filename);
    
    return fetch(url)
        .then(response => response.arrayBuffer())
        .then(buffer => new File([buffer], filename, { type: mimeType }));
}

// Handle selected element and convert it to File object.
function selectItem(event) {
    if(control==1){
        window.alert("Kontrol işlemi başladığı için resim seçilemez. İşlem bitince yeni resim seçebilirsiniz.");
        return;
    }
    const selectedImageSrc = event.target.querySelectorAll('img')[0].src;
    const img = document.getElementById('radiograph');
    const iconContainer = document.getElementById('iconCont');
    const filename = event.target.getAttribute('file_name');
    
    img.src = selectedImageSrc;
    img.style.display = 'block';
    iconContainer.style.display = 'none';

    // Convert selected image URL to File object
    imageURLtoFile(selectedImageSrc, filename)
        .then(file => {
            //console.log('Converted File:', file);
            imageReference = file;
        })
        .catch(error => {
            console.error('Error converting image to file object:', error);
        });
}


// prevent image icon from drag.
function preventIconDrag(event){
    event.preventDefault();
}

// to handle dropping image to container.
function handleImageDropOperation(event) {
    if(control==1){
        window.alert("Kontrol işlemi başladığı için resim seçilemez. İşlem bitince yeni resim seçebilirsiniz.");
        return;
    }
    event.preventDefault();
    if (event.dataTransfer.files[0].type.startsWith('image/')) {        // Accept only image files.
        loadImage(event.dataTransfer.files[0]);
    }else{
        window.alert("Bu alana sadece resim yükleyebilirsiniz.");
    }
    event.dataTransfer.clearData();
}

// create toast and show massage to user.
function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.display = 'inline';
    setTimeout(function() {toast.style.display = 'none';}, 3000);
}