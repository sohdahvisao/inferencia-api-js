const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./swagger.json');
const express = require('express');
const fs = require('fs');

const app = express()
const port = 3000
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

let customModel = null;
let featureExtractor = null;
let parameters = null;
let modelJson = null;


app.use(express.text({ type: 'text/plain', limit: '10mb' }));

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

app.get('/', (req, res) => {
    res.send('Classification API Working')
})

app.post('/classification-image', async (req, res) => {
    if (!req.body || req.body.length === 0)
        return res.status(400).send('No image data received.');

    try {
        // Converte o texto base64 de volta em um Buffer de bytes
        const imageBuffer = Buffer.from(req.body, 'base64');
        const image = tfn.node.decodeImage(imageBuffer, 3);
        const result = await predictImage(image);
        tfn.dispose(image);
        res.json(result);
    } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).send('Error processing image');
    }
});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})

loadMobileNetFeatureModel();
loadLocalModel()

async function loadMobileNetFeatureModel() {
    // Caminho do diretório onde o modelo MobileNet está localizado
    const mobileNetPath = 'graph_model/mobilenet/model.json';
    const handler = tfn.io.fileSystem(mobileNetPath)
    featureExtractor = await tf.loadGraphModel(handler);
    console.log('MobileNet Feature Model carregado com sucesso!');

    tf.tidy(function () {
        let answer = featureExtractor.predict(
            tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
        );
        console.log(answer.shape);
    });
}

async function loadLocalModel() {
    const modelJsonPath = 'soda_vision_model/model.json';
    const weightsPath = 'soda_vision_model/model.weights.bin';
    const parametersPath = 'soda_vision_model/parameters.json';
    let modelWeightsBuffer = null;
    try {
        modelJson = JSON.parse(fs.readFileSync(modelJsonPath).toString());
        parameters = JSON.parse(fs.readFileSync(parametersPath).toString());
        modelWeightsBuffer = fs.readFileSync(weightsPath)

        console.log('Modelo carregado com sucesso!');

        customModel = await tf.models.modelFromJSON(modelJson);

        const weightSpecs = modelJson.weightSpecs;
        const weightData = new Uint8Array(modelWeightsBuffer);
        const tensors = tf.io.decodeWeights(weightData, weightSpecs);
        customModel.setWeights(Object.values(tensors));

    } catch (error) {
        console.error('Erro ao carregar o modelo:', error);
    }
}

function calculateFeaturesOnCurrentFrame(image) {
    return tf.tidy(function () {
        // const videoFrameAsTensor = tf.browser.fromPixels(image);
        const resizedTensorFrame = tf.image.resizeBilinear(
            image,
            [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
            true
        );

        const normalizedTensorFrame = resizedTensorFrame.div(255);

        return featureExtractor
            .predict(normalizedTensorFrame.expandDims())
            .squeeze();
    });
}

async function predictImage(rawImage) {
    if (!customModel || !featureExtractor)
        return;
    return tf.tidy(() => {
        const imageFeatures =
            calculateFeaturesOnCurrentFrame(rawImage);
        const prediction = customModel
            .predict(imageFeatures.expandDims())
            .squeeze();
        const highestIndex = prediction.argMax().arraySync();
        const predictionArray = prediction.arraySync();

        const predictedClass = parameters.classNames[highestIndex];
        const predictedConfidence = Math.floor(
            predictionArray[highestIndex] * 100
        );
        return { "classification": predictedClass, "confidence-score": predictedConfidence }
    });
}