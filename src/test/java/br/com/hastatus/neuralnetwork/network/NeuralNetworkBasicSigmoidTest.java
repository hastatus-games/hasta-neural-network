package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.train.TrainStopCondition;
import br.com.hastatus.neuralnetwork.train.backpropagation.TrainBackPropagation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuralNetworkBasicSigmoidTest {

    @Test
    public void identityFunctionTest() {

        int inputSize = 1;
        int[] hiddenSizes = {10};
        int outputSize = 1;
        NeuralNetwork neuralNetwork = new NeuralNetworkBasicSigmoid(inputSize, hiddenSizes, outputSize);

        double[][] sampleInput = {{0}, {0.1}, {0.15}, {0.2}, {0.25}, {0.3}, {0.35}, {0.4},{0.45},  {0.5},{0.55},  {0.6},{0.65},  {0.7},{0.75},  {0.8},{0.85},  {0.9},{0.95},  {1.0}};
        double[][] expectedOutput = {{0}, {0.1}, {0.15}, {0.2}, {0.25}, {0.3}, {0.35}, {0.4},{0.45},  {0.5},{0.55},  {0.6},{0.65},  {0.7},{0.75},  {0.8},{0.85},  {0.9},{0.95},  {1.0}};


        TrainBackPropagation trainBackPropagation = new TrainBackPropagation(neuralNetwork);

        TrainStopCondition trainStopCondition = new TrainStopCondition(10000, 0.1, true, 0.0012);
        trainBackPropagation.train(sampleInput, expectedOutput, trainStopCondition);

        final double tolerance = 0.08;


        double[] inputs1 = {0.1};
        double[] outputs1 = neuralNetwork.feedForward(inputs1);
        Assertions.assertEquals(0.1, outputs1[0], tolerance);


        double[] inputs2 = {0.5};
        double[] outputs2 = neuralNetwork.feedForward(inputs2);
        Assertions.assertEquals(0.5, outputs2[0], tolerance);


        double[] inputs3 = {1};
        double[] outputs3 = neuralNetwork.feedForward(inputs3);
        Assertions.assertEquals(1, outputs3[0], tolerance);
    }


}
