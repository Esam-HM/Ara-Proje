// load image
document.getElementById('uploadBtn').addEventListener('click',function(){
    document.getElementById('input').click();
    loadImage();
    imageListener();
});






// checkBtn clicked.
document.getElementById('checkBtn').addEventListener('click', function() {
    const progressBar = document.getElementById('progress');
    const img = document.getElementById('radiograph');
    const inputImg = document.getElementById('input').files[0];
    const resultCont = document.getElementById('resContId');
    const inImage = document.getElementById('inputImg');
    const mand = document.getElementById('mask');
    const overlay= document.getElementById('overlayMask');
    const uploadCont = document.getElementById('uploadContId');
    const formData = new FormData();


    removeImageListener();

    if(img.getAttribute('src') == ''){
        showToast('Load image');
        return;
    }
    // uploadCont.style.display = 'none';
    // resultCont.style.display = 'flex';

    progressBar.style.display = 'flex';

    formData.append('rontgen',inputImg);

    fetch("/uploadImage", {
        method: "POST",
        body: formData,
    }).then(response => {
        if (!response.ok) {
            console.log('Error happend when getting response');
        }
        return response.json();
    }).then(data => {
        console.log(data.response);
        inImage.src = img.getAttribute('src');
        mand.src = `data:;base64,${data.mandibula}`;
        overlay.src = `data:;base64,${data.overlayMask}`;
        progressBar.style.display = 'none';
        uploadCont.style.display = 'none';
        resultCont.style.display = 'flex';
        console.log('Process finished');
    }).catch(error => {
        console.error('Error Happend', error);
    });
});

document.getElementById('backBtn').addEventListener('click',function(){
    const resCont = document.getElementById('resContId');
    const uploadCont = document.getElementById('uploadContId');

    deleteImage();
    resCont.style.display = 'none';
    uploadCont.style.display = 'flex';

});

// delete loaded image
function imageListener(){
    document.getElementById('radiograph').addEventListener('dblclick', deleteImage);
}

// prevent delete image
function removeImageListener(){
    document.getElementById('radiograph').removeEventListener('dblclick', deleteImage);
}

function loadImage(){
    document.getElementById('input').addEventListener('change', function() {
        const file = this.files[0];
        const reader = new FileReader();
        const img = document.getElementById('radiograph');
        const iconContainer = document.getElementById('iconCont');
        
        reader.onload = function(e) {
            iconContainer.style.display = 'none';
            img.style.display = 'block'
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.display = 'inline';
    
    setTimeout(function() {toast.style.display = 'none';}, 3000);
}

function deleteImage(){
    const img = document.getElementById('radiograph');
    const iconContainer = document.getElementById('iconCont');
    const progressBar = document.getElementById('progress');
    img.setAttribute('src', '');
    iconContainer.style.display = 'flex';

    if(progressBar.style.display != 'none'){
        progressBar.style.display = 'none';
    }
}