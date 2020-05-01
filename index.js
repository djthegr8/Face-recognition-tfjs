// Simply initialising the variables
let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var p1Samples=0, p2Samples=0, p3Samples=0;
let isPredicting = false;

// This function is UI-related
function submitNames(){
    for(i = 0;i < 3;i++){
        if(document.getElementById("name"+i).value == ""){
            alert("Please enter names")
            window.location.reload();
        }
    }
    for(i = 0;i <= 2;i++){
        document.getElementById(i).innerHTML = document.getElementById("name"+i).value;
        document.getElementById("name"+i).readOnly = true;
    }
}
// Loads the MobileNet
async function loadMobilenet() {
  const mobilenets = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenets.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenets.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 50,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			p1Samples++;
			document.getElementById("person1samples").innerText = document.getElementById("name0").value + " samples:" + p1Samples;
			break;
		case "1":
			p2Samples++;
			document.getElementById("person2samples").innerText = document.getElementById("name1").value + " samples:" + p2Samples;
			break;
		case "2":
			p3Samples++;
			document.getElementById("person3samples").innerText = document.getElementById("name2").value + " samples:" + p3Samples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see " + document.getElementById("name0").value;
			break;
		case 1:
			predictionText = "I see " + document.getElementById("name1").value;
			break;
		case 2:
			predictionText = "I see " + document.getElementById("name2").value;
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
    console.log("Training over. You are good to go.")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
