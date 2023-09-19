package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.train.backpropagation.TrainBackPropagation;
import org.junit.jupiter.api.Test;

public class NeuralNetworkBasicSigmoidTest {

    @Test
    public void calculatorTest() {

        int inputSize = 1;
        int[] hiddenSizes = {10};
        int outputSize = 1;
        NeuralNetwork neuralNetwork = new NeuralNetworkBasicSigmoid(inputSize, hiddenSizes, outputSize);

        double[][] sampleInput = {{0}, {0.1}, {0.15}, {0.2}, {0.25}, {0.3}, {0.35}, {0.4},{0.45},  {0.5},{0.55},  {0.6},{0.65},  {0.7},{0.75},  {0.8},{0.85},  {0.9},{0.95},  {1.0}};
        double[][] expectedOutput = {{0}, {0.1}, {0.15}, {0.2}, {0.25}, {0.3}, {0.35}, {0.4},{0.45},  {0.5},{0.55},  {0.6},{0.65},  {0.7},{0.75},  {0.8},{0.85},  {0.9},{0.95},  {1.0}};


        TrainBackPropagation trainBackPropagation = new TrainBackPropagation(neuralNetwork);

        trainBackPropagation.train(sampleInput, expectedOutput, 10000, 0.01, 0.001);


        System.out.println(" -------  end ------ \n\n");

        double[] inputs1 = {0.1};
        double[] outputs1 = neuralNetwork.feedForward(inputs1);
        System.out.println("Output for 0.1: " + outputs1[0]);

        double[] inputs2 = {0.5};
        double[] outputs2 = neuralNetwork.feedForward(inputs2);
        System.out.println("Output for 0.5: " + outputs2[0]);


        double[] inputs3 = {1};
        double[] outputs3 = neuralNetwork.feedForward(inputs3);
        System.out.println("Output for 1: " + outputs3[0]);
    }


}
