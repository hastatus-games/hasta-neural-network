package br.com.hastatus.neuralnetwork.train.backpropagation;

import br.com.hastatus.neuralnetwork.base.Neuron;
import br.com.hastatus.neuralnetwork.base.NeuronSigmoid;
import br.com.hastatus.neuralnetwork.layer.NeuralLayer;
import br.com.hastatus.neuralnetwork.network.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class TrainBackPropagation {

    private static final Logger logger = LogManager.getLogger(TrainBackPropagation.class);
    private TrainConvergencyTracker trainConvergencyTracker;

    final private NeuralNetwork neuralNetwork;

    public TrainBackPropagation(NeuralNetwork neuralNetwork){
        this.neuralNetwork = neuralNetwork;
    }


    public void train(double[][] trainingInputs, double[][] expectedOutputs, int numEpochs, double learningRate) {

        trainConvergencyTracker = new TrainConvergencyTracker();

        for (int epoch = 0; epoch < numEpochs; epoch++) {

            logger.debug(" *** Training Epoch: {} / {}", epoch, numEpochs-1);

            for (int i = 0; i < trainingInputs.length; i++) {
                logger.debug("Training for sample inputs[{}]", i);
                // Forward pass
                double[] output = neuralNetwork.feedForward(trainingInputs[i]);

                double[] expectedOutput = expectedOutputs[i];

                TrainingDeltas trainingDeltas = getTrainingDeltas(trainingInputs[i], output, expectedOutput);
                adjustWeightsAndBias(trainingDeltas, learningRate);
            }

        }
    }


    private TrainingDeltas getTrainingDeltas(double[] inputs, double[] outputs, double[] expected) {
        if(outputs.length!=expected.length) {
            throw new RuntimeException("The number of total values expected cant be different from the network output ");
        }

        TrainingDeltas trainingDeltas = new TrainingDeltas(inputs, neuralNetwork.getTotalLayers());

        trainingDeltas.initDeltaLayer(neuralNetwork.getTotalLayers()-1, expected.length);

        //iterate over values and calculate output
        for (int j = 0; j < outputs.length; j++) {

            double error = outputs[j] - expected[j];


            trainConvergencyTracker.addMeanSquareError(error); //calculate the Mean Square Error to track the convergency (if the training is converging to expected results)


            double delta = error * NeuronSigmoid.deriveSigmoid(outputs[j]);

            trainingDeltas.addDelta(neuralNetwork.getTotalLayers()-1, j, delta);
            logger.debug("Output[{}/{}]: {} expected:{} error:{} trainingDelta:{}", j, outputs.length-1, outputs[j], expected[j], error, delta);

        }

        // Calculate deltas for the hidden layers
        for (int j = neuralNetwork.getTotalLayers() - 2; j >= 0; j--) {
            NeuralLayer layer = neuralNetwork.getLayer(j);
            NeuralLayer nextLayer = neuralNetwork.getLayer(j + 1);
            double[] inputNeuronio = layer.getLastInputs();

            trainingDeltas.initDeltaLayer(j, layer.getTotalNeurons());

            for (int k = 0; k < layer.getTotalNeurons(); k++) {
                double sum = 0;
                for (int l = 0; l < nextLayer.getTotalNeurons(); l++) {
                    sum += nextLayer.getNeuron(l).getWeight(k) * trainingDeltas.getDelta(j+1, l);
                }


                double activation = layer.getNeuron(k).activate(inputNeuronio);
                double delta = sum * NeuronSigmoid.deriveSigmoid(activation);
                trainingDeltas.addDelta(j, k, delta);
            }
        }

        // Update weights and biases
        return trainingDeltas;
    }



    private void adjustWeightsAndBias(TrainingDeltas trainingDeltas, double learningRate) {
        for (int j = 0; j < neuralNetwork.getTotalLayers(); j++) {

            double[] inputsFromLayer;
            if(j == 0) {
                inputsFromLayer = trainingDeltas.getInputs();
            } else {
                inputsFromLayer = neuralNetwork.getLayer(j-1).getLastInputs();
            }


            for (int i = 0; i < neuralNetwork.getLayer(j).getTotalNeurons(); i++) {
                Neuron neuron = neuralNetwork.getLayer(j).getNeuron(i);

                neuron.adjustBias(trainingDeltas.getDelta(j, i) * learningRate);

                for (int l = 0; l < inputsFromLayer.length; l++) {
                    neuron.adjustWeight(l, trainingDeltas.getDelta(j, i) * inputsFromLayer[l] * learningRate);
                }

            }
        }
    }
}
