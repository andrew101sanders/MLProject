function onFileSelected(event) {
  var selectedFile = event.target.files[0];
  var reader = new FileReader();

  let imgtag = document.getElementById("image-display");

  reader.onload = function (event) {
    imgtag.src = event.target.result;
    predictImage(imgtag.src);
  };

  reader.readAsDataURL(selectedFile);
}

function predictImage(image) {
  console.log("predicting...");
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(image),
  })
    .then((response) => response.json())
    .then((data) => {
      displayResult(data.prediction);
      console.log("Success:", data);
    })
    .catch((err) => {
      console.log("An error occured", err.message);
    });
}

function displayResult(data) {
  console.log("predicted!");
  const predtag = document.getElementById("prediction");
  predtag.innerHTML = data;
}
