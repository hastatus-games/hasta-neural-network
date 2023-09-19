package br.com.hastatus.neuralnetwork.train.backpropagation;

import br.com.hastatus.neuralnetwork.base.Neuron;
import br.com.hastatus.neuralnetwork.base.NeuronSigmoid;
import br.com.hastatus.neuralnetwork.layer.NeuralLayer;
import br.com.hastatus.neuralnetwork.network.NeuralNetwork;
import br.com.hastatus.neuralnetwork.train.NeuralNetworkTrain;
import br.com.hastatus.neuralnetwork.train.TrainStopCondition;
import br.com.hastatus.neuralnetwork.train.TrainingDeltas;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class TrainBackPropagation implements NeuralNetworkTrain  {

    private static final Logger logger = LogManager.getLogger(TrainBackPropagation.class);
    private TrainConvergencyTracker trainConvergencyTracker;

    final private NeuralNetwork neuralNetwork;

    public TrainBackPropagation(NeuralNetwork neuralNetwork){
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Performs the training of the neural network using the backpropagation algorithm. The training is carried out
     * iteratively through epochs until a stopping condition, specified by {@link TrainStopCondition}, is met.
     *
     * @param trainingInputs A 2D array where each row represents a set of inputs to the network.
     * @param expectedOutputs A 2D array where each row represents the set of expected outputs from the network for the corresponding set of inputs.
     * @param trainStopCondition An object encapsulating the conditions under which the training should be halted. This includes criteria such as a maximum number of epochs, a defined learning rate, an indication of whether to stop in case of divergence, and an acceptable mean square error threshold.
     */
    public void train(double[][] trainingInputs, double[][] expectedOutputs, TrainStopCondition trainStopCondition) {

        trainConvergencyTracker = new TrainConvergencyTracker();

        int totalEpochs = trainStopCondition.getMaxEpochs();

        for (int epoch = 0; epoch < totalEpochs; epoch++) {

            logger.trace(" *** Training Epoch: {} / {}", epoch, totalEpochs-1);

            for (int i = 0; i < trainingInputs.length; i++) {

                logger.trace("Training for sample inputs[{}]", i);
                // Forward pass
                double[] output = neuralNetwork.feedForward(trainingInputs[i]);

                double[] expectedOutput = expectedOutputs[i];

                TrainingDeltas trainingDeltas = getTrainingDeltas(trainingInputs[i], output, expectedOutput);
                adjustWeightsAndBias(trainingDeltas, trainStopCondition.getLearningRate());
            }

            trainConvergencyTracker.calculateAndSaveMeanSquareError(trainingInputs.length, expectedOutputs[0].length);

            if(epoch > 0) {
                double currentMSE = trainConvergencyTracker.getMeanSquareErrorByEpoch(epoch);
                double previousMSE = trainConvergencyTracker.getMeanSquareErrorByEpoch(epoch-1);
                double evolution = previousMSE - currentMSE;

                if(currentMSE <= trainStopCondition.getMeanSquareErrorGood()) {
                    logger.info("EPOCH[{}/{}] MeanSquareErrorAcceptable: {}", epoch, totalEpochs, currentMSE);
                    break;
                }

                if(evolution < 0) {
                    logger.warn("EPOCH[{}/{}] Training not converging: {}", epoch, totalEpochs, evolution);

                    if(trainStopCondition.isStopOnDiverge()) {
                        break;
                    }
                    break;
                }
                else {
                    logger.debug("EPOCH[{}/{}] currentMSE: {}, previousMSE: {}  Evolution: {}", epoch, totalEpochs, currentMSE, previousMSE, evolution);
                }
            }
        }
    }



    private TrainingDeltas getTrainingDeltas(double[] inputs, double[] outputs, double[] expected) {
        if(outputs.length!=expected.length) {
            throw new RuntimeException("The number of total values expected can't be different from the network output ");
        }

        TrainingDeltas trainingDeltas = new TrainingDeltas(inputs, neuralNetwork.getTotalLayers());

        trainingDeltas.initDeltaLayer(neuralNetwork.getTotalLayers()-1, expected.length);

        //iterate over values and calculate output
        for (int j = 0; j < outputs.length; j++) {

            double error = outputs[j] - expected[j];


            trainConvergencyTracker.addMeanSquareError(error); //calculate the Mean Square Error to track the convergency (if the training is converging to expected results)


            double delta = error * NeuronSigmoid.deriveSigmoid(outputs[j]);

            trainingDeltas.addDelta(neuralNetwork.getTotalLayers()-1, j, delta);
            logger.trace("Output[{}/{}]: {} expected:{} error:{} trainingDelta:{}", j, outputs.length-1, outputs[j], expected[j], error, delta);

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
