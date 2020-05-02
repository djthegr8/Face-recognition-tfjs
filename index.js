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

// Loads the MobileNet and removes final layer
async function loadMobilenet() {
  const mobilenets = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenets.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenets.inputs, outputs: layer.output});
}


//This function trains the model on the newly made dataset
async function train() {
  // Initializes a new empty dataset with labels
  dataset.ys = null;
  // Specifies the number of labels as 3 (Three people)
  dataset.encodeLabels(3);
  // A new Sequential() model with a Flatten layer and two Dense layers.
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  // Using ADAM as the optimizer
  const optimizer = tf.train.adam(0.0001);
  // Specifying the loss as the categorical cross-entropy
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  // Initializing loss to zero
  let loss = 0;
  // Fitting the model on the images in the dataset for 50 epochs while printing the loss every time
  model.fit(dataset.xs, dataset.ys, {
    epochs: 50,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
      },
      onEpochEnd: async (epoch,logs) => {
          if(epoch == 49){
              console.log("Training over. You are good to go");
          }
      }
        
      }
   });
}


/*
This function allows labelling and effectively creating the dataset.
Also, the count of number of samples per user is maintained
*/
function handleButton(elem){
	switch(elem.id){
		case "0":
			// This will run if the user presses person-1's button
			// The value will be incremented and displayed
			p1Samples++;
			document.getElementById("person1samples").innerText = document.getElementById("name0").value + " samples:" + p1Samples;
			break;
		case "1":
			// Similarly for person-2
			p2Samples++;
			document.getElementById("person2samples").innerText = document.getElementById("name1").value + " samples:" + p2Samples;
			break;
		case "2":
			// And for person-3
			p3Samples++;
			document.getElementById("person3samples").innerText = document.getElementById("name2").value + " samples:" + p3Samples;
			break;
	}
  // gives the value 0, 1 or 2 to the label
	label = parseInt(elem.id);
  // gets the image from the webcam
	const img = webcam.capture();
  // adds both, the img and label to the dataset
	dataset.addExample(mobilenet.predict(img), label);

}

// A function to start prediction
async function predict() {
  // isPredicting is true when the Start Predicting button has been pressed
  while (isPredicting) {
    // tf.tidy() simply clears unused variables  
    const predictedClass = tf.tidy(() => {
      // webcam is a class that is from a premade Google jS page
      const img = webcam.capture();
      // Finds encodings for MobileNet
      const activation = mobilenet.predict(img);
      // Predicts
      const predictions = model.predict(activation);
      //.as1D.argMax() finds the value with highest probability	     
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    // The below switch(){} outputs the prediction's name	  
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
			
    // dispose clears the predictedClass variable
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

// UI handler
function doTraining(){
	train();
}

// Ensures that prediction is started once
function startPredicting(){
	isPredicting = true;
	predict();
}

// Stops predicting when pressed
function stopPredicting(){
	isPredicting = false;
	predict();
}

// Combines everything
async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
