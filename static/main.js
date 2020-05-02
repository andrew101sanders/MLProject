let CNNPrediction = 0;
let RegressionPrediction = 0;

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
      CNNPrediction = parseFloat(data.prediction);
      console.log("Success:", data);
    })
    .catch((err) => {
      console.log("An error occured", err.message);
    });
}

function onSubmit() {
  let param1 = parseInt(document.getElementById("param1").value) || 0;
  let param2 = parseInt(document.getElementById("param2").value) || 0;
  let param3 = parseInt(document.getElementById("param3").value) || 0;
  let amenities = Array.from(
    document.querySelectorAll(".amenity")
  ).map((amenity) => (amenity.checked ? 1 : 0));

  console.log(amenities);
  console.log("regressing...");
  fetch("/regress", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify([param1, param2, param3, ...amenities]),
  })
    .then((response) => response.json())
    .then((data) => {
      RegressionPrediction = parseFloat(data.regressRes);
      displayResult();
      console.log("Success:", data);
    })
    .catch((err) => {
      console.log("An error occured", err.message);
    });
}

function displayResult() {
  const prediction =
    Math.round((0.9 * RegressionPrediction + 0.1 * CNNPrediction) * 100) / 100;
  const predtag = document.getElementById("prediction");
  predtag.innerHTML = prediction;
}

function onCheckChange(event) {
  let label = document.getElementById(event.target.id + "label");
  label.classList.contains("checked")
    ? label.classList.remove("checked")
    : label.classList.add("checked");
}
